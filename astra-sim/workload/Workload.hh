/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_HH__
#define __WORKLOAD_HH__

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Statistics.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/LocalMemUsageTracker.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

namespace AstraSim {

class Sys;
class DataSet;

class Workload : public Callable {
  public:
    Workload(Sys* sys,
             std::string et_filename,
             std::string comm_group_filename);
    ~Workload();

    // communicator groups
    // Parse the user provided 'comm_group_filename' and extract the list of
    // communicator groups. Refer to the wiki for the format.
    void initialize_comm_groups(std::string comm_group_filename);
    void issue_pytorch_pg_metadata(
        std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);

    // event-based simulation
    void issue_dep_free_nodes();
    void issue(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_metadata(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_replay(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_remote_mem(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_comp(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_coll_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_send_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_recv_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    // Stage 4 (collectives-evolve), opt-in through node attributes; absent = unchanged behaviour.
    //   proxy_net (uint64, 1)   network op: counted as active on this rank's proxy while issued
    //   proxy_svc_ns            FIFO occupancy of the rank's proxy: a send waits for it before injection,
    //                           a receive after its data arrived
    //   proxy_inject_ns         send only: injection time after the FIFO service starts (default = svc)
    //   proxy_poll_ns           receive only: dynamic cost poll x (active network ops on the rank - 1),
    //                           paid after the data arrived (receive-side progress over active ops)
    //   comm_post_delay_ns      the node completes this much after its event (step work)
    //   comm_delay_ns on a receive: posting delay (Stage 1 traces always carry 0 there)
    void send_now(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node, Tick delay);
    void proxy_enqueue_send(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    Tick proxy_busy_until = 0;
    int proxy_active = 0;
    // Stage 4b (collectives-evolve), opt-in through node attributes; absent = unchanged behaviour.
    //   xtb_bind_ns, xtb_chk_ns, xtb_parent (K')   a step with a cross-TB dependency starts at
    //                           max(t_ctrl + xtb_chk_ns, finish(xtb_parent) + xtb_bind_ns) (send: added to the send
    //                           delay; receive: added to the posting delay); see xtb_edge_delay()
    //   recv_work_ns (W)        receive only: finish = max(event + post delay + poll, posting time + recv_work_ns)
    Tick xtb_edge_delay(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) const;
    // M1 (collectives-evolve), opt-in through node attributes; absent = unchanged behaviour.
    //   duration_ns (COMP node)  compute time in ns; replaces duration_micros x 1000 (0 -> 1 ns, as before)
    //   handoff_depth (send)     the send runs in hand-off mode on its connection (src, dst, tag), which holds
    //                            handoff_depth slots (MSCCL runtime: NCCL_STEPS = 8). After its send delay the send
    //                            waits until handoff_slots slots are free (FIFO per connection), takes them and
    //                            injects its data; with handoff_slots > 0 the node finishes max(1, handoff_ns) later (data
    //                            handed to the connection buffer / proxy), not at arrival. handoff_slots = 0: the
    //                            node finishes at arrival (old rule) but is still queued on the connection.
    //   handoff_release (recv)   when this receive finishes (the receiver consumed the data), the slots of the
    //                            matching in-flight send (same connection, oldest with the same byte count) are
    //                            freed and blocked sends of that connection proceed.
    //   proxy_active (Stage 4 poll): a hand-off send stays active on its rank until its data arrives.
    void handoff_acquire(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void handoff_start(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void handoff_release(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void handoff_arrived(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    std::unordered_map<uint64_t, Tick> finish_tick;
    std::unordered_map<uint64_t, Tick> recv_post_tick;
    // Stage 5 (collectives-evolve), opt-in through node attributes; absent = unchanged behaviour.
    // One proxy thread per rank is a single FIFO server (s5_proxy_busy). Network sends with s5_net take a
    // separate path: send delay -> slots on the connection (handoff_depth / handoff_slots) -> GPU work -> proxy
    // -> per-connection pipeline -> injection -> arrival -> proxy completion work -> slots freed (SENDER side).
    //   s5_net (send/recv)       1: network op served by this rank's proxy
    //   s5_px_pre_ns             proxy work booked when the op is posted (send: injection waits for it;
    //                            receive: booked at posting, nothing waits for it)
    //   s5_px_post_ns            proxy work booked at the data's arrival (send: the slots are freed after it;
    //                            receive: see s5_px_wait)
    //   s5_px_wait (recv)        1: the receive completes after its post work (Simple: the GPU waits for the proxy's
    //                            tail update); 0: the GPU sees the data at arrival (LL flags)
    //   s5_work_ns (send)        GPU send-step work after the slots are taken; a hand-off send (handoff_slots > 0)
    //                            finishes after it; handoff_slots = 0: the node finishes at its data's arrival
    //   s5_post_after_ns (send)  time after the slots are taken at which the op reaches the proxy (<= s5_work_ns)
    //   s5_px_bg_ns              proxy work of the other FIFO slots of a multi-slot op, booked right after the posting
    //                            work (send) or at posting (receive); nothing waits for it, but the gate below does
    //   s5_conn_ns (send)        connection pace time of the data (P / per-connection rate); the connection
    //                            (src, dst, tag) is a FIFO: the data starts at max(proxy pre work end, connection free)
    //   s5_conn_occ_ns (send)    occupancy of the connection if longer than the pace (per-message serialization)
    //   s5_inject_ns (send)      injection delay after the connection start (fixed fabric latency)
    //   s5_route_ns (send)       link latency of the route (to place the gate at the receiver)
    //   gate: the data counts as arrived (sender completion and receiver) at max(network arrival,
    //         inject + route + s5_conn_ns, end of the sender's background work + inject delay + route,
    //         end of the receiver's background work)
    //   s5_rx_lat_ns (recv)      receive latency after the proxy's completion work (Simple GDR flush), with s5_px_wait
    //   comm_post_delay_ns (recv) the step's own work after its data is visible. A receive posted after its data
    //                            was complete (gate passed) sees the data at max(posting, gate + completion + latency):
    //                            the proxy completes receives on its own, independent of the GPU step
    void s5_after(Tick delay, std::function<void()> fn);
    Tick s5_proxy_book(Tick work);
    void s5_finish_later(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node, Tick delay);
    void s5_send_acquire(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void s5_send_go(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void s5_send_post(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void s5_send_arrived(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void s5_release(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void s5_recv_arrived(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node, WorkloadLayerHandlerData* wlhd);
    Tick s5_proxy_busy = 0;
    // Stage 5b: per-rank LL data server. s5_rank_ns (send): this send's LL data time on the rank's one server; it
    // starts at max(proxy posting end, server free) and the gate also waits for its end + inject delay + route
    Tick s5_rank_busy = 0;
    std::unordered_map<uint64_t, Tick> s5_send_gate;    // send node -> gate tick
    std::unordered_map<uint64_t, Tick> s5_recv_bg_end;  // receive node -> end of its background proxy work
    void skip_invalid(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void call(EventType event, CallData* data);
    void fire();

    // stats
    void report();

    Chakra::ETFeeder* et_feeder;
    std::unordered_map<int, CommunicatorGroup*> comm_groups;
    HardwareResource* hw_resource;
    Sys* sys;
    Statistics* stats;
    std::unique_ptr<LocalMemUsageTracker> local_mem_usage_tracker;
    std::unordered_map<int, uint64_t> collective_comm_node_id_map;
    std::unordered_map<int, DataSet*> collective_comm_wrapper_map;
    bool is_finished;

    // completion check (krydzy/check-complete): every ET node must finish.
    // total_nodes = node messages in this rank's ET file; finished_nodes counts finish_node calls.
    // A rank can report "finished" with nodes left (no node ready or in flight: a dependency cycle), or never report
    // (a posted recv that no send matches). check_complete() logs either case; main() then exits non-zero unless
    // ASTRASIM_ALLOW_INCOMPLETE=1 (the old behaviour).
    uint64_t total_nodes = 0;
    uint64_t finished_nodes = 0;
    bool check_complete() const;

  private:
    // From the ET node, find out the corresponding communicator group, and
    // return the pointer. If no communicator group is specified for this ET
    // node, return nullptr.
    CommunicatorGroup* extract_comm_group(
        std::shared_ptr<Chakra::ETFeederNode> node);
};

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
