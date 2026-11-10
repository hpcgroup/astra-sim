/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/Workload.hh"

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/MemEventHandlerData.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include <json/json.hpp>

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <tuple>
#include <unistd.h>

using namespace std;
using namespace AstraSim;
using namespace Chakra::FeederV3;
using json = nlohmann::json;

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

// completion check (krydzy/check-complete): number of node messages in an ET file. The file is a sequence of
// varint32-length-prefixed protobuf messages (feeder_v3/protobuf_util.h); the first one is the global metadata.
// Only the framing is read (no parsing): one sequential pass with seeks.
static uint64_t count_et_nodes(const string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::in);
    uint64_t messages = 0;
    while (true) {
        uint32_t size = 0;
        int shift = 0;
        bool ok = false;
        uint8_t byte;
        while (f.read(reinterpret_cast<char*>(&byte), 1)) {
            size |= static_cast<uint32_t>(byte & 0x7f) << shift;
            if (!(byte & 0x80)) {
                ok = true;
                break;
            }
            shift += 7;
            if (shift > 28) {
                break;
            }
        }
        if (!ok) {
            break;
        }
        f.seekg(size, std::ios::cur);
        if (!f) {
            break;
        }
        messages++;
    }
    return messages > 0 ? messages - 1 : 0;
}

Workload::Workload(Sys* sys, string et_filename, string comm_group_filename) {
    string workload_filename = et_filename + "." + to_string(sys->id) + ".et";
    // Check if workload filename exists
    if (access(workload_filename.c_str(), R_OK) < 0) {
        string error_msg;
        if (errno == ENOENT) {
            error_msg =
                "workload file: " + workload_filename + " does not exist";
        } else if (errno == EACCES) {
            error_msg = "workload file: " + workload_filename +
                        " exists but is not readable";
        } else {
            error_msg =
                "Unknown workload file: " + workload_filename + " access error";
        }
        LoggerFactory::get_logger("workload")->critical(error_msg);
        exit(EXIT_FAILURE);
    }
    this->et_feeder = new ETFeeder(workload_filename);
    this->total_nodes = count_et_nodes(workload_filename);
    this->comm_groups.clear();
    // TODO: parametrize the number of available hardware resources
    this->hw_resource = new HardwareResource(1, sys->id);
    this->local_mem_usage_tracker =
        std::make_unique<LocalMemUsageTracker>(sys->id);
    this->sys = sys;
    initialize_comm_groups(comm_group_filename);
    this->stats = new Statistics(this);
    this->is_finished = false;
}

Workload::~Workload() {
    for (auto comm_group : comm_groups) {
        delete comm_group.second;
    }
    comm_groups.clear();

    if (this->et_feeder != nullptr) {
        delete this->et_feeder;
    }
    if (this->hw_resource != nullptr) {
        delete this->hw_resource;
    }
    if (this->stats != nullptr) {
        delete this->stats;
    }
}

void Workload::initialize_comm_groups(string comm_group_filename) {
    // communicator group input file is not given
    if (comm_group_filename.find("empty") != std::string::npos) {
        comm_groups.clear();
        return;
    }

    ifstream inFile;
    json j;
    inFile.open(comm_group_filename);
    inFile >> j;

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        std::string comm_group_name = it.key();
        int comm_group_id = std::stoi(comm_group_name);

        std::vector<int> involved_NPUs;
        for (auto id : it.value()) {
            involved_NPUs.push_back(id);
        }

        comm_groups[comm_group_id] =
            new CommunicatorGroup(comm_group_id, involved_NPUs, sys);
    }
}

void Workload::issue_pytorch_pg_metadata(
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    // For read comm groups from torch, might overwrite previous.
    std::string pg_info = node->get_inputs_values();
    if (pg_info.empty()) {
        return;
    }
    pg_info = pg_info.substr(2, pg_info.size() - 4);

    try {
        json valuesRoot = json::parse(pg_info);

        for (const auto& item : valuesRoot) {
            std::string pgName = item.at("pg_name").get<std::string>();
            std::vector<int> involved_NPUs =
                item.at("ranks").get<std::vector<int>>();

            if (involved_NPUs.empty()) {
                for (int i = 0; i < sys->total_nodes; i++) {
                    involved_NPUs.push_back(i);
                }
            }

            int32_t pgNameInt = std::stoi(pgName);
            // To ensure pgName > 0
            CommunicatorGroup* cg =
                new CommunicatorGroup(pgNameInt + 1, involved_NPUs, sys);
            this->comm_groups[pgNameInt] = cg;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing or processing JSON: " << e.what()
                  << std::endl;
    }
}

void Workload::issue_dep_free_nodes() {
    auto& dependancy_resolver = this->et_feeder->getDependancyResolver();
    auto dependancy_free_nodes =
        dependancy_resolver.get_dependancy_free_nodes();
    std::set<uint64_t> dependancy_free_nodes_set;
    for (const auto node_id : dependancy_free_nodes) {
        dependancy_free_nodes_set.insert(node_id);
    }
    for (const auto node_id : dependancy_free_nodes_set) {
        std::shared_ptr<ETFeederNode> node = et_feeder->lookupNode(node_id);
        if (hw_resource->is_available(node)) {
            issue(node);
        }
    }
}

void Workload::issue(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    auto logger = LoggerFactory::get_logger("workload");
    if (sys->trace_enabled) {
        logger->debug("issue,sys->id={}, tick={}, node->id={}, "
                      "node->name={}, node->type={}",
                      sys->id, Sys::boostedTick(), node->id(), node->name(),
                      static_cast<uint64_t>(node->type()));
    }

    this->et_feeder->getDependancyResolver().take_node(node->id());
    this->hw_resource->occupy(node);
    // stats->record_end will be called in Workload::call
    stats->record_start(node, Sys::boostedTick());
    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->recordStart(node, Sys::boostedTick());
    }
    if (sys->replay_only) {
        issue_replay(node);
    } else {
        if ((node->type() == ChakraNodeType::MEM_LOAD_NODE) ||
            (node->type() == ChakraNodeType::MEM_STORE_NODE)) {
            issue_remote_mem(node);
        } else if (node->type() == ChakraNodeType::COMP_NODE) {
            if (!this->sys->roofline_enabled) {
                issue_replay(node);
            } else {
                if (node->is_cpu_op<bool>(false)) {
                    // comp node on cpu
                    // should only appears in real system trace and should run
                    // with replay.
                    issue_replay(node);
                } else {
                    // comp node on gpu
                    issue_comp(node);
                }
            }
        } else if (node->type() == ChakraNodeType::COMM_COLL_NODE ||
                   node->type() == ChakraNodeType::COMM_SEND_NODE ||
                   node->type() == ChakraNodeType::COMM_RECV_NODE) {
            issue_comm(node);
        } else if (node->type() == ChakraNodeType::INVALID_NODE) {
            skip_invalid(node);
        } else if (node->type() == ChakraNodeType::METADATA_NODE) {
            issue_metadata(node);
        } else {
            logger->critical("Unknown node type");
            exit(EXIT_FAILURE);
        }
    }
}

void Workload::issue_metadata(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    // TODO: someway to identify this metadata node is a pytorch pg node
    if (true) {
        issue_pytorch_pg_metadata(node);
    } else {
        throw std::runtime_error("Unknown metadata node type");
    }
    this->skip_invalid(node);  // for proper dependancy resolving
}

void Workload::issue_replay(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node->id();
    uint64_t runtime = 1ul;
    if (node->has_attr("duration_ns")) {
        // M1 (opt-in): compute time carried in ns
        const auto ns = node->get_attr<uint64_t>("duration_ns", 0);
        runtime = ns != 0ul ? ns : 1ul;
    } else if (node->runtime() != 0ul) {
        // chakra runtimes are in microseconds and we should convert it into
        // nanoseconds
        runtime = node->runtime() * 1000;
    }
    if (node->is_cpu_op()) {
        hw_resource->tics_cpu_ops += runtime;
    } else {
        hw_resource->tics_gpu_ops += runtime;
    }
    sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_remote_mem(
    shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node->id();
    sys->remote_mem->issue(node->tensor_size(), wlhd);
}

void Workload::issue_comp(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (!this->sys->roofline_enabled) {
        throw std::runtime_error(
            "Roofline model is not enabled for non-replay comp");
    }

    if (node->is_cpu_op()) {
        throw std::runtime_error("Roofline is only available for GPU nodes");
        return;
    }

    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node->id();

    double num_ops = static_cast<double>(node->num_ops<uint64_t>());
    double tensor_size = static_cast<double>(node->tensor_size<uint64_t>());

    // if tensor_size is 0 during roofline mode, this is an invalid node
    if (tensor_size == 0) {
        skip_invalid(node);
        return;
    }

    double operational_intensity = num_ops / tensor_size;
    double perf = sys->roofline->get_perf(operational_intensity);
    double elapsed_time = static_cast<double>(node->num_ops()) / perf;  // sec
    uint64_t runtime = static_cast<uint64_t>(elapsed_time * 1e9);  // sec -> ns
    if (node->is_cpu_op()) {
        hw_resource->tics_cpu_ops += runtime;
    } else {
        hw_resource->tics_gpu_ops += runtime;
    }
    sys->register_event(this, EventType::General, wlhd, runtime);

    auto& op_stat = this->stats->get_operator_statistics(node->id());
    op_stat.operation_intensity = operational_intensity;
    op_stat.compute_utilization = perf / sys->peak_perf;
    op_stat.memory_utilization =
        (perf / operational_intensity) / sys->local_mem_bw;
    op_stat.is_memory_bound = perf < sys->peak_perf;
    LoggerFactory::get_logger("workload")
        ->debug("operation_intensity={}, perf={}, elapsed_time={} "
                "compute_utilization={} memory_utilization={} tensor_size={} "
                "num_ops={}",
                operational_intensity, perf, elapsed_time,
                op_stat.compute_utilization.value(),
                op_stat.memory_utilization.value(), tensor_size, num_ops);
}

void Workload::issue_comm(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->is_cpu_op<bool>(false)) {
        throw std::runtime_error("Comm node should not be on CPU");
    }
    const auto node_type = node->type();
    if (node_type == ChakraNodeType::COMM_COLL_NODE) {
        this->issue_coll_comm(node);
    } else if (node_type == ChakraNodeType::COMM_SEND_NODE) {
        this->issue_send_comm(node);
    } else if (node_type == ChakraNodeType::COMM_RECV_NODE) {
        this->issue_recv_comm(node);
    } else {
        throw std::runtime_error("Unknown comm node type");
    }
}

void Workload::issue_coll_comm(
    shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const bool has_involve_dims = node->has_attr("involve_dims");
    std::vector<bool> involved_dims;
    if (node->has_attr("involved_dim")) {
        const ChakraProtoMsg::AttributeProto& attr =
            node->get_attr_msg("involved_dim");

        // Ensure the attribute is of type bool_list before accessing
        if (attr.has_bool_list()) {
            const ChakraProtoMsg::BoolList& bool_list = attr.bool_list();

            // Traverse bool_list and add values to involved_dim
            for (int i = 0; i < bool_list.values_size(); ++i) {
                involved_dims.push_back(bool_list.values(i));
            }
        } else {
            cerr << "Expected bool_list in involved_dim but found another type."
                 << endl;
            exit(EXIT_FAILURE);
        }
    } else {
        // involved_dims does not exist in ETFeeder.
        // Assume involved_dims = [1,1,1,1,1] which we could simulate
        // 5-Dimension. Could use Process Group to build involved_dims later.
        // Once process group is implemented, you should get
        // that with node->pg_name()

        for (int i = 0; i < 4; i++) {
            involved_dims.push_back(true);
        }
    }

    CommunicatorGroup* comm_group = extract_comm_group(node);
    const auto comm_type =
        static_cast<ChakraCollectiveCommType>(node->comm_type<uint64_t>());
    const auto comm_size = node->comm_size<uint64_t>();
    // Record communication size for bandwidth calculation
    stats->get_operator_statistics(node->id()).comm_size = comm_size;
    // TODO: comm_tag? which is used to distinguish two different collective in
    // same pg
    const auto comm_priority = node->comm_priority<uint32_t>();  // default 0u

    if (comm_type == ChakraCollectiveCommType::ALL_REDUCE) {
        DataSet* fp = sys->generate_all_reduce(comm_size, involved_dims,
                                               comm_group, comm_priority, node->id());
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::ALL_TO_ALL) {
        DataSet* fp = sys->generate_all_to_all(comm_size, involved_dims,
                                               comm_group, comm_priority, node->id());
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::ALL_GATHER) {
        DataSet* fp = sys->generate_all_gather(comm_size, involved_dims,
                                               comm_group, comm_priority, node->id());
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::REDUCE_SCATTER) {
        DataSet* fp = sys->generate_reduce_scatter(comm_size, involved_dims,
                                                   comm_group, comm_priority, node->id());
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::BROADCAST) {
        // TODO: implement broadcast, for now just replay
        uint64_t runtime = 1ul;
        if (node->runtime() != 0ul) {
            // chakra runtimes are in microseconds and we should convert it into
            // nanoseconds
            runtime = node->runtime() * 1000;
        }
        DataSet* fp = new DataSet(1);
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        sys->register_event(fp, EventType::General, nullptr,
                            // chakra runtimes are in microseconds and we
                            // should convert it into nanoseconds
                            runtime);
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else {
        throw std::runtime_error("Unsupported collective comm type");
    }
}

namespace {
// Deferred FIFO enqueue of a send on the rank's proxy (Stage 4).
class ProxySendCaller : public Callable {
  public:
    ProxySendCaller(Workload* w, shared_ptr<Chakra::FeederV3::ETFeederNode> n)
        : workload(w), node(std::move(n)) {}
    void call(EventType, CallData*) override {
        workload->proxy_enqueue_send(node);
        delete this;
    }
    Workload* workload;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node;
};
}  // namespace

void Workload::proxy_enqueue_send(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const Tick now = Sys::boostedTick();
    const Tick svc = node->get_attr<uint64_t>("proxy_svc_ns", 0);
    const Tick inject = node->get_attr<uint64_t>("proxy_inject_ns", svc);
    const Tick start = std::max(now, proxy_busy_until);
    proxy_busy_until = start + svc;
    send_now(node, start - now + inject);
}

// Stage 4b K' (opt-in): a step with a cross-TB dependency starts at max(t_ctrl + chk, t_prod + bind), where t_prod
// is the finish of the cross-TB parent `xtb_parent` and t_ctrl the latest finish of the other parents. The node is
// issued when its last parent finished (now). If the producer finished last, t_ctrl + chk <= now + chk <= now + bind
// (chk <= bind, checked by the converter), so the delay is bind; otherwise now = t_ctrl. Both cases:
// delay = max(chk, t_prod + bind - now).
Tick Workload::xtb_edge_delay(shared_ptr<Chakra::FeederV3::ETFeederNode> node) const {
    const Tick bind = node->get_attr<uint64_t>("xtb_bind_ns", 0);
    if (bind == 0) {
        return 0;
    }
    const Tick chk = node->get_attr<uint64_t>("xtb_chk_ns", 0);
    const uint64_t parent = node->get_attr<uint64_t>("xtb_parent", 0);
    const auto it = finish_tick.find(parent);
    if (it == finish_tick.end()) {
        throw std::runtime_error("xtb_parent of node " + std::to_string(node->id()) + " has not finished");
    }
    const Tick now = Sys::boostedTick();
    const Tick prod = it->second + bind;
    return std::max(chk, prod > now ? prod - now : 0);
}

// Stage 5 (opt-in, see Workload.hh). Connection state is shared by all ranks (one process runs every rank).
namespace {
class S5Fn : public Callable {
  public:
    explicit S5Fn(std::function<void()> f) : fn(std::move(f)) {}
    void call(EventType, CallData*) override {
        auto f = std::move(fn);
        delete this;
        f();
    }
    std::function<void()> fn;
};
struct S5Waiting {
    Workload* workload;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node;
};
struct S5Conn {
    uint64_t used = 0;              // slots taken (handoff_depth per connection)
    std::deque<S5Waiting> waiting;  // sends blocked on slots, FIFO
    Tick busy = 0;                  // the connection pipeline is busy until this tick
};
std::map<std::tuple<int, int, int>, S5Conn> s5_conns;
// Stage 5c: per-NIC duplex budget, one FIFO server per rank's NIC shared by its sends and its receives (all ranks)
std::map<int, Tick> s5_nic_busy;
// gate tick of each in-flight Stage 5 message, per (src, dst, tag, bytes) in send order (= AstraSim's matching order)
std::map<std::tuple<int, int, int, uint64_t>, std::deque<Tick>> s5_gates;

std::tuple<int, int, int> s5_key(int src, shared_ptr<Chakra::FeederV3::ETFeederNode> const& node) {
    return std::make_tuple(src, static_cast<int>(node->comm_dst<uint32_t>()), static_cast<int>(node->comm_tag<uint32_t>()));
}

// Receives the network's send callback (the data arrived) of a Stage 5 send.
class S5Arrival : public Callable {
  public:
    S5Arrival(Workload* w, shared_ptr<Chakra::FeederV3::ETFeederNode> n) : workload(w), node(std::move(n)) {}
    void call(EventType, CallData* data) override {
        delete static_cast<WorkloadLayerHandlerData*>(data);
        auto* w = workload;
        auto n = node;
        delete this;
        w->s5_send_arrived(n);
    }
    Workload* workload;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node;
};
}  // namespace

void Workload::s5_after(Tick delay, std::function<void()> fn) {
    // a 0-delay event registered while the network processes its current event list trips EventQueue::proceed's
    // assertion (see the M1 hand-off), so a zero delay runs the function now
    if (delay == 0) {
        fn();
        return;
    }
    sys->register_event(new S5Fn(std::move(fn)), EventType::General, nullptr, delay);
}

Tick Workload::s5_proxy_book(Tick work) {
    const Tick now = Sys::boostedTick();
    if (work == 0) {
        return now;
    }
    const Tick start = std::max(now, s5_proxy_busy);
    s5_proxy_busy = start + work;
    return s5_proxy_busy;
}

void Workload::s5_finish_later(shared_ptr<Chakra::FeederV3::ETFeederNode> node, Tick delay) {
    auto* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node->id();
    sys->register_event(this, EventType::General, wlhd, std::max<Tick>(1, delay));
}

void Workload::s5_send_acquire(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->comm_src<uint32_t>(this->sys->id) != this->sys->id) {
        throw std::runtime_error("Send node should be issued by the sender");
    }
    const auto depth = node->get_attr<uint64_t>("handoff_depth", 0);
    const auto slots = node->get_attr<uint64_t>("handoff_slots", 0);
    if (depth > 0 && slots > 0) {
        if (slots > depth) {
            throw std::runtime_error("handoff_slots > handoff_depth on node " + std::to_string(node->id()));
        }
        auto& conn = s5_conns[s5_key(sys->id, node)];
        if (!conn.waiting.empty() || conn.used + slots > depth) {
            conn.waiting.push_back(S5Waiting{this, node});
            return;
        }
        conn.used += slots;
    }
    s5_send_go(node);
}

void Workload::s5_send_go(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const Tick work = node->get_attr<uint64_t>("s5_work_ns", 0);
    const Tick post_after = std::min(work, static_cast<Tick>(node->get_attr<uint64_t>("s5_post_after_ns", 0)));
    if (node->get_attr<uint64_t>("handoff_depth", 0) > 0 && node->get_attr<uint64_t>("handoff_slots", 0) > 0) {
        s5_finish_later(node, work);  // the GPU step ends when its data is in the connection's slots
    }
    s5_after(post_after, [this, node]() { s5_send_post(node); });
}

void Workload::s5_send_post(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const Tick now = Sys::boostedTick();
    const Tick pre_end = s5_proxy_book(node->get_attr<uint64_t>("s5_px_pre_ns", 0));
    // background proxy work of the other FIFO slots (pipelined with the stream): FIFO occupancy, no one waits
    const Tick bg = node->get_attr<uint64_t>("s5_px_bg_ns", 0);
    const Tick bg_end = bg > 0 ? s5_proxy_book(bg) : pre_end;
    auto& conn = s5_conns[s5_key(sys->id, node)];
    const Tick cstart = std::max(pre_end, conn.busy);
    const Tick pace = node->get_attr<uint64_t>("s5_conn_ns", 0);
    conn.busy = cstart + std::max(pace, static_cast<Tick>(node->get_attr<uint64_t>("s5_conn_occ_ns", 0)));
    const Tick inject_ns = node->get_attr<uint64_t>("s5_inject_ns", 0);
    const Tick inject_at = cstart + inject_ns;
    const Tick route = node->get_attr<uint64_t>("s5_route_ns", 0);
    // the last byte cannot arrive before the connection pipeline and the proxy's per-slot work are done
    Tick gate = std::max(inject_at + route + pace, bg_end + inject_ns + route);
    // Stage 5b: per-rank LL data server (s5_rank_ns), shared by every LL network send of this rank, FIFO
    const Tick rank_pace = node->get_attr<uint64_t>("s5_rank_ns", 0);
    if (rank_pace > 0) {
        const Tick rstart = std::max(pre_end, s5_rank_busy);
        s5_rank_busy = rstart + rank_pace;
        gate = std::max(gate, s5_rank_busy + inject_ns + route);
    }
    // Stage 5c: the wire bytes also take the sender's and the receiver's NIC duplex server (s5_nic_src_ns / _dst_ns)
    const Tick nic_src = node->get_attr<uint64_t>("s5_nic_src_ns", 0);
    const Tick nic_dst = node->get_attr<uint64_t>("s5_nic_dst_ns", 0);
    if (nic_src > 0 || nic_dst > 0) {
        const int peer = static_cast<int>(node->comm_dst<uint32_t>());
        auto& bs = s5_nic_busy[static_cast<int>(sys->id)];
        bs = std::max(pre_end, bs) + nic_src;
        auto& bd = s5_nic_busy[peer];
        bd = std::max(pre_end, bd) + nic_dst;
        gate = std::max(gate, std::max(bs, bd) + inject_ns + route);
    }
    s5_send_gate[node->id()] = gate;
    const auto dst = node->comm_dst<uint32_t>();
    const auto size = node->comm_size<uint64_t>();
    const auto tag = node->comm_tag<uint32_t>();
    s5_gates[std::make_tuple(static_cast<int>(sys->id), static_cast<int>(dst), static_cast<int>(tag), size)].push_back(gate);
    stats->get_operator_statistics(node->id()).comm_size = size;
    sim_request snd_req;
    snd_req.srcRank = sys->id;
    snd_req.dstRank = dst;
    snd_req.reqType = UINT8;
    SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
    sehd->callable = new S5Arrival(this, node);
    sehd->wlhd = new WorkloadLayerHandlerData;
    sehd->wlhd->node_id = node->id();
    sehd->event = EventType::PacketSent;
    sys->front_end_sim_send(inject_at - now, Sys::dummy_data, size, UINT8, dst, tag, &snd_req,
                            Sys::FrontEndSendRecvType::NATIVE, &Sys::handleEvent, sehd);
}

void Workload::s5_send_arrived(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const Tick now = Sys::boostedTick();
    Tick gate = now;
    const auto it = s5_send_gate.find(node->id());
    if (it != s5_send_gate.end()) {
        gate = std::max(now, it->second);
        s5_send_gate.erase(it);
    }
    s5_after(gate - now, [this, node]() {
        const Tick t = Sys::boostedTick();
        const Tick end = s5_proxy_book(node->get_attr<uint64_t>("s5_px_post_ns", 0));
        const bool handed_off =
            node->get_attr<uint64_t>("handoff_depth", 0) > 0 && node->get_attr<uint64_t>("handoff_slots", 0) > 0;
        if (!handed_off) {
            s5_finish_later(node, 1);  // the node finishes when its data arrived
            return;
        }
        s5_after(end - t, [this, node]() { s5_release(node); });
    });
}

void Workload::s5_recv_arrived(shared_ptr<Chakra::FeederV3::ETFeederNode> node, WorkloadLayerHandlerData* wlhd) {
    const Tick now = Sys::boostedTick();
    Tick t_data = 0;  // the message is complete: its gate (sender side) and this receive's background proxy work
    const auto key = std::make_tuple(static_cast<int>(node->comm_src<uint32_t>()), static_cast<int>(sys->id),
                                     static_cast<int>(node->comm_tag<uint32_t>()), node->comm_size<uint64_t>());
    auto g = s5_gates.find(key);
    if (g != s5_gates.end() && !g->second.empty()) {
        t_data = g->second.front();  // matching is in send order per (src, dst, tag, bytes), as AstraSim's
        g->second.pop_front();
    }
    // this receive's own background proxy work (other FIFO slots) gates only a step posted before its data was
    // complete; the proxy posts a kernel's receives at launch, independent of when the GPU step starts
    Tick t_wait = t_data;
    const auto b = s5_recv_bg_end.find(node->id());
    if (b != s5_recv_bg_end.end()) {
        t_wait = std::max(t_wait, b->second);
        s5_recv_bg_end.erase(b);
    }
    const Tick post = node->get_attr<uint64_t>("s5_px_post_ns", 0);
    const Tick rx_lat = node->get_attr<uint64_t>("s5_rx_lat_ns", 0);
    const Tick tail = node->get_attr<uint64_t>("comm_post_delay_ns", 0);
    const bool wait = node->get_attr<uint64_t>("s5_px_wait", 0) != 0;
    if (now <= t_data) {
        // the step was posted before its data was complete: when the data and this receive's background work are
        // done the proxy completes it (FIFO work), the receive latency (Simple flush) follows, then the step's work
        s5_after(t_wait - now, [this, wlhd, post, rx_lat, tail, wait]() {
            const Tick t2 = Sys::boostedTick();
            const Tick end = s5_proxy_book(post);
            const Tick visible = wait ? end + rx_lat : t2;
            sys->register_event(this, EventType::General, wlhd, std::max<Tick>(1, visible - t2 + tail));
        });
        return;
    }
    // the step was posted after its data was complete: the proxy completed (and flushed) it on its own when the data
    // landed. Its work is booked now for the FIFO occupancy; the step waits only for what is left of the latency.
    s5_proxy_book(post);
    const Tick visible = wait ? std::max(now, t_data + post + rx_lat) : now;
    sys->register_event(this, EventType::General, wlhd, std::max<Tick>(1, visible - now + tail));
}


void Workload::s5_release(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    auto& conn = s5_conns[s5_key(sys->id, node)];
    conn.used -= node->get_attr<uint64_t>("handoff_slots", 0);
    while (!conn.waiting.empty()) {
        const auto front = conn.waiting.front();
        const auto slots = front.node->get_attr<uint64_t>("handoff_slots", 0);
        if (conn.used + slots > front.node->get_attr<uint64_t>("handoff_depth", 0)) {
            break;
        }
        conn.waiting.pop_front();
        conn.used += slots;
        front.workload->s5_send_go(front.node);
    }
}

void Workload::issue_send_comm(
    shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->get_attr<uint64_t>("s5_net", 0)) {
        const Tick delay = node->comm_delay_ns<uint64_t>() + xtb_edge_delay(node);
        s5_after(delay, [this, node]() { s5_send_acquire(node); });
        return;
    }
    if (node->get_attr<uint64_t>("proxy_net", 0)) {
        proxy_active++;
    }
    const auto delay = node->comm_delay_ns<uint64_t>() + xtb_edge_delay(node);
    if (node->get_attr<uint64_t>("proxy_svc_ns", 0) > 0) {
        auto* caller = new ProxySendCaller(this, node);
        if (delay == 0) {
            caller->call(EventType::General, nullptr);
        } else {
            sys->register_event(caller, EventType::General, nullptr, delay);
        }
        return;
    }
    send_now(node, delay);
}

// M1 send hand-off (opt-in, see Workload.hh). One slot table per connection (src, dst, tag), shared by all ranks
// (the analytical frontend runs every rank's Workload in one process).
namespace {
struct HandoffPending {
    Workload* workload;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node;
    uint64_t slots;
};
struct HandoffConn {
    uint64_t used = 0;
    std::deque<std::pair<uint64_t, uint64_t>> inflight;  // (bytes, slots) per started send, in start order
    std::deque<HandoffPending> waiting;                   // sends blocked on slots, FIFO
};
std::map<std::tuple<int, int, int>, HandoffConn> handoff_conns;

class HandoffCaller : public Callable {
  public:
    HandoffCaller(Workload* w, shared_ptr<Chakra::FeederV3::ETFeederNode> n) : workload(w), node(std::move(n)) {}
    void call(EventType, CallData*) override {
        workload->handoff_acquire(node);
        delete this;
    }
    Workload* workload;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node;
};

// Receives the network's send callback (data arrived) of a hand-off send whose node already finished.
class HandoffArrival : public Callable {
  public:
    HandoffArrival(Workload* w, shared_ptr<Chakra::FeederV3::ETFeederNode> n) : workload(w), node(std::move(n)) {}
    void call(EventType, CallData* data) override {
        workload->handoff_arrived(node);
        delete static_cast<WorkloadLayerHandlerData*>(data);
        delete this;
    }
    Workload* workload;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node;
};
}  // namespace

void Workload::handoff_acquire(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const auto key = std::make_tuple(static_cast<int>(sys->id), static_cast<int>(node->comm_dst<uint32_t>()),
                                     static_cast<int>(node->comm_tag<uint32_t>()));
    auto& conn = handoff_conns[key];
    const auto depth = node->get_attr<uint64_t>("handoff_depth", 0);
    const auto slots = node->get_attr<uint64_t>("handoff_slots", 0);
    if (slots > depth) {
        throw std::runtime_error("handoff_slots > handoff_depth on node " + std::to_string(node->id()));
    }
    if (!conn.waiting.empty() || conn.used + slots > depth) {
        conn.waiting.push_back(HandoffPending{this, node, slots});
        return;
    }
    handoff_start(node);
}

void Workload::handoff_start(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const auto dst = node->comm_dst<uint32_t>();
    const auto tag = node->comm_tag<uint32_t>();
    const auto size = node->comm_size<uint64_t>();
    const auto slots = node->get_attr<uint64_t>("handoff_slots", 0);
    auto& conn = handoff_conns[std::make_tuple(static_cast<int>(sys->id), static_cast<int>(dst), static_cast<int>(tag))];
    conn.used += slots;
    conn.inflight.emplace_back(size, slots);
    stats->get_operator_statistics(node->id()).comm_size = size;

    sim_request snd_req;
    snd_req.srcRank = sys->id;
    snd_req.dstRank = dst;
    snd_req.reqType = UINT8;
    SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
    sehd->wlhd = new WorkloadLayerHandlerData;
    sehd->wlhd->node_id = node->id();
    sehd->event = EventType::PacketSent;
    if (slots > 0) {
        // the node finishes at hand-off; the network's callback only marks the arrival
        sehd->callable = new HandoffArrival(this, node);
        auto* done = new WorkloadLayerHandlerData;
        done->node_id = node->id();
        // at least 1 ns: a 0-delay event registered while the network's current event list is being processed
        // trips EventQueue::proceed's assertion (same floor as a 0-duration COMP node)
        const Tick handoff_ns = std::max<Tick>(1, node->get_attr<uint64_t>("handoff_ns", 0));
        sys->register_event(this, EventType::PacketSent, done, handoff_ns);
    } else {
        sehd->callable = this;  // old rule: the node finishes when its data arrives
    }
    sys->front_end_sim_send(0, Sys::dummy_data, size, UINT8, dst, tag, &snd_req, Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, sehd);
}

void Workload::handoff_arrived(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->get_attr<uint64_t>("proxy_net", 0)) {
        proxy_active--;
    }
}

void Workload::handoff_release(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const auto key = std::make_tuple(static_cast<int>(node->comm_src<uint32_t>()), static_cast<int>(sys->id),
                                     static_cast<int>(node->comm_tag<uint32_t>()));
    const auto it = handoff_conns.find(key);
    if (it == handoff_conns.end()) {
        return;  // the sender of this connection does not run in hand-off mode
    }
    auto& conn = it->second;
    const auto size = node->comm_size<uint64_t>();
    for (auto e = conn.inflight.begin(); e != conn.inflight.end(); ++e) {
        if (e->first == size) {
            conn.used -= e->second;
            conn.inflight.erase(e);
            break;
        }
    }
    while (!conn.waiting.empty()) {
        const auto& front = conn.waiting.front();
        const auto depth = front.node->get_attr<uint64_t>("handoff_depth", 0);
        if (conn.used + front.slots > depth) {
            break;
        }
        const auto pending = front;
        conn.waiting.pop_front();
        pending.workload->handoff_start(pending.node);
    }
}

void Workload::send_now(shared_ptr<Chakra::FeederV3::ETFeederNode> node, Tick delay) {
    if (node->get_attr<uint64_t>("handoff_depth", 0) > 0) {
        if (node->comm_src<uint32_t>(this->sys->id) != this->sys->id) {
            throw std::runtime_error("Send node should be issued by the sender");
        }
        auto* caller = new HandoffCaller(this, node);
        if (delay == 0) {
            caller->call(EventType::General, nullptr);
        } else {
            sys->register_event(caller, EventType::General, nullptr, delay);
        }
        return;
    }
    const auto src = node->comm_src<uint32_t>(this->sys->id);
    if (src != this->sys->id) {
        throw std::runtime_error("Send node should be issued by the sender");
    }
    const auto dst = node->comm_dst<uint32_t>();
    const auto size = node->comm_size<uint64_t>();
    // Record communication size for bandwidth calculation
    stats->get_operator_statistics(node->id()).comm_size = size;
    const auto tag = node->comm_tag<uint32_t>();

    sim_request snd_req;
    snd_req.srcRank = src;
    snd_req.dstRank = dst;
    snd_req.reqType = UINT8;
    SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
    sehd->callable = this;
    sehd->wlhd = new WorkloadLayerHandlerData;
    sehd->wlhd->node_id = node->id();
    sehd->event = EventType::PacketSent;
    sys->front_end_sim_send(delay, Sys::dummy_data, size, UINT8, dst, tag,
                            &snd_req, Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, sehd);
}

void Workload::issue_recv_comm(
    shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const auto src = node->comm_src<uint32_t>();
    const auto dst = node->comm_dst<uint32_t>(this->sys->id);
    if (dst != this->sys->id) {
        throw std::runtime_error("Recv node should be issued by the receiver");
    }
    const auto size = node->comm_size<uint64_t>();
    // Record communication size for bandwidth calculation
    stats->get_operator_statistics(node->id()).comm_size = size;
    const auto tag = node->comm_tag<uint32_t>();
    if (node->get_attr<uint64_t>("proxy_net", 0)) {
        proxy_active++;
    }
    const Tick post_delay = node->get_attr<uint64_t>("comm_delay_ns", 0) + xtb_edge_delay(node);
    if (node->get_attr<uint64_t>("s5_net", 0)) {
        s5_proxy_book(node->get_attr<uint64_t>("s5_px_pre_ns", 0));  // Stage 5: the proxy posts the receive
        const Tick bg = node->get_attr<uint64_t>("s5_px_bg_ns", 0);
        if (bg > 0) {
            s5_recv_bg_end[node->id()] = s5_proxy_book(bg);  // per-slot work of the other FIFO slots
        }
    }
    if (node->get_attr<uint64_t>("recv_work_ns", 0) > 0) {
        recv_post_tick[node->id()] = Sys::boostedTick() + post_delay;
    }

    sim_request rcv_req;
    RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
    rcehd->wlhd = new WorkloadLayerHandlerData;
    rcehd->wlhd->node_id = node->id();
    rcehd->workload = this;
    rcehd->event = EventType::PacketReceived;
    sys->front_end_sim_recv(post_delay, Sys::dummy_data, size, UINT8, src, tag, &rcv_req,
                            Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, rcehd);
}

void Workload::skip_invalid(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const auto node_id = node->id();
    auto& dependancy_resolver = this->et_feeder->getDependancyResolver();
    dependancy_resolver.finish_node(node_id);
    this->finished_nodes++;
    finish_tick[node_id] = Sys::boostedTick();
    auto logger = LoggerFactory::get_logger("workload");
    logger->debug("callback,sys->id={}, tick={}, node->id={}, "
                  "node->name={}, node->type={}",
                  sys->id, Sys::boostedTick(), node->id(), node->name(),
                  static_cast<uint64_t>(node->type()));
    hw_resource->release(node);
    stats->record_end(node, Sys::boostedTick());
    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->recordEnd(node, Sys::boostedTick());
    }
}

void Workload::call(EventType event, CallData* data) {
    if (is_finished) {
        return;
    }

    if (event == EventType::CollectiveCommunicationFinished) {
        IntData* int_data = (IntData*)data;
        uint64_t coll_comm_id = int_data->data;

        hw_resource->tics_gpu_comms += int_data->execution_time;
        uint64_t node_id = collective_comm_node_id_map[coll_comm_id];
        shared_ptr<Chakra::FeederV3::ETFeederNode> node =
            et_feeder->lookupNode(node_id);

        if (sys->trace_enabled) {
            LoggerFactory::get_logger("workload")
                ->debug("callback,sys->id={}, tick={}, node->id={}, "
                        "node->name={}, node->type={}",
                        sys->id, Sys::boostedTick(), node->id(), node->name(),
                        static_cast<uint64_t>(node->type()));
        }

        hw_resource->release(node);
        stats->record_end(node, Sys::boostedTick());

        // Calculate network bandwidth
        auto& op_stat = stats->get_operator_statistics(node_id);
        Tick execution_time = int_data->execution_time;
        if (execution_time > 0 && op_stat.comm_size.has_value()) {
            double bandwidth =
                static_cast<double>(op_stat.comm_size.value()) / execution_time;
            op_stat.network_bandwidth = bandwidth;
        }

        if (this->sys->track_local_mem) {
            this->local_mem_usage_tracker->recordEnd(node, Sys::boostedTick());
        }

        this->et_feeder->getDependancyResolver().finish_node(node_id);
        this->finished_nodes++;
        finish_tick[node_id] = Sys::boostedTick();

        issue_dep_free_nodes();

        // The Dataset class provides statistics that should be used later to
        // dump more statistics in the workload layer
        delete collective_comm_wrapper_map[coll_comm_id];
        collective_comm_wrapper_map.erase(coll_comm_id);

    } else {
        if (data == nullptr) {
            issue_dep_free_nodes();
        } else {
            WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
            shared_ptr<Chakra::FeederV3::ETFeederNode> node =
                et_feeder->lookupNode(wlhd->node_id);

            // Stage 5: a network receive waits for its gate, its background slots and its proxy completion work
            if (event == EventType::PacketReceived && node->get_attr<uint64_t>("s5_net", 0)) {
                s5_recv_arrived(node, wlhd);
                return;
            }
            // Stage 4: receive-side proxy cost and step work after the event (opt-in attributes).
            if (event == EventType::PacketSent || event == EventType::PacketReceived) {
                const Tick now = Sys::boostedTick();
                Tick extra = node->get_attr<uint64_t>("comm_post_delay_ns", 0);
                if (event == EventType::PacketReceived) {
                    const Tick svc = node->get_attr<uint64_t>("proxy_svc_ns", 0);
                    if (svc > 0) {
                        const Tick start = std::max(now, proxy_busy_until);
                        proxy_busy_until = start + svc;
                        extra += start + svc - now;
                    }
                    const Tick poll = node->get_attr<uint64_t>("proxy_poll_ns", 0);
                    if (poll > 0 && proxy_active > 1) {
                        extra += poll * static_cast<Tick>(proxy_active - 1);
                    }
                    // Stage 4b W (opt-in): the receive's step work starts at its posting time and overlaps the
                    // wait for data: finish = max(event + post delay + poll, posted + recv_work_ns).
                    const Tick work = node->get_attr<uint64_t>("recv_work_ns", 0);
                    if (work > 0) {
                        const auto it = recv_post_tick.find(node->id());
                        if (it == recv_post_tick.end()) {
                            throw std::runtime_error("recv_work_ns: no posting time for node " + std::to_string(node->id()));
                        }
                        const Tick done = it->second + work;
                        if (done > now + extra) {
                            extra = done - now;
                        }
                        recv_post_tick.erase(it);
                    }
                }
                if (extra > 0) {
                    sys->register_event(this, EventType::General, wlhd, extra);
                    return;
                }
            }
            const bool handoff_early = node->type() == ChakraNodeType::COMM_SEND_NODE &&
                                       node->get_attr<uint64_t>("handoff_depth", 0) > 0 &&
                                       node->get_attr<uint64_t>("handoff_slots", 0) > 0;
            if ((node->type() == ChakraNodeType::COMM_SEND_NODE ||
                 node->type() == ChakraNodeType::COMM_RECV_NODE) &&
                node->get_attr<uint64_t>("proxy_net", 0) && !handoff_early) {
                proxy_active--;  // a hand-off send leaves the proxy at its data's arrival (handoff_arrived)
            }
            if (node->type() == ChakraNodeType::COMM_RECV_NODE && node->get_attr<uint64_t>("handoff_release", 0)) {
                handoff_release(node);
            }

            if (sys->trace_enabled) {
                LoggerFactory::get_logger("workload")
                    ->debug("callback,sys->id={}, tick={}, node->id={}, "
                            "node->name={}, node->type={}",
                            sys->id, Sys::boostedTick(), node->id(),
                            node->name(), static_cast<uint64_t>(node->type()));
            }

            hw_resource->release(node);
            stats->record_end(node, Sys::boostedTick());

            // Calculate network bandwidth for point-to-point communications
            if (event == EventType::PacketSent ||
                event == EventType::PacketReceived) {
                auto& op_stat = stats->get_operator_statistics(wlhd->node_id);
                Tick execution_time =
                    stats->get_operator_statistics(wlhd->node_id).end_time -
                    stats->get_operator_statistics(wlhd->node_id).start_time;
                if (execution_time > 0 && op_stat.comm_size.has_value()) {
                    double bandwidth =
                        static_cast<double>(op_stat.comm_size.value()) /
                        execution_time;
                    op_stat.network_bandwidth = bandwidth;
                }
            }

            if (this->sys->track_local_mem) {
                this->local_mem_usage_tracker->recordEnd(node,
                                                         Sys::boostedTick());
            }

            this->et_feeder->getDependancyResolver().finish_node(wlhd->node_id);
            this->finished_nodes++;
            finish_tick[wlhd->node_id] = Sys::boostedTick();

            issue_dep_free_nodes();

            delete wlhd;
        }
    }

    const auto& dep_resolver = this->et_feeder->getDependancyResolver();
    if ((dep_resolver.get_dependancy_free_nodes().empty()) &&
        (dep_resolver.get_ongoing_nodes().empty()) &&
        (hw_resource->num_in_flight_cpu_ops == 0) &&
        (hw_resource->num_in_flight_gpu_comp_ops == 0) &&
        (hw_resource->num_in_flight_gpu_comm_ops == 0)) {
        report();
        sys->comm_NI->sim_notify_finished();
        is_finished = true;
    }
}

bool Workload::check_complete() const {
    if (is_finished && finished_nodes == total_nodes) {
        return true;
    }
    LoggerFactory::get_logger("workload")
        ->critical("sys[{}] INCOMPLETE: {} of {} ET nodes finished ({}): the ET cannot run to completion "
                   "(dependency cycle or unmatched send/recv); its wall time is not a result.",
                   sys->id, finished_nodes, total_nodes,
                   is_finished ? "reported finished with nodes left" : "never reported finished");
    return false;
}

void Workload::fire() {
    call(EventType::General, NULL);
}

void Workload::report() {
    Tick curr_tick = Sys::boostedTick();
    LoggerFactory::get_logger("workload")
        ->info("sys[{}] finished, {} cycles, exposed communication {} cycles.",
               sys->id, curr_tick, curr_tick - hw_resource->tics_gpu_ops);
    stats->post_processing();
    stats->report();
    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->buildMemoryTrace();
        this->local_mem_usage_tracker->buildMemoryTimeline();
        this->local_mem_usage_tracker->dumpMemoryTrace(
            this->sys->local_mem_trace_filename);
        auto [peak_mem_usage, unit] =
            this->local_mem_usage_tracker->getPeakMemUsageFormatted();
        auto logger = LoggerFactory::get_logger("workload");
        logger->info("sys[{}] peak memory usage: {:.2f} {}", sys->id,
                     peak_mem_usage, unit);
        this->local_mem_usage_tracker.reset();
    }
}

CommunicatorGroup* Workload::extract_comm_group(
    std::shared_ptr<Chakra::ETFeederNode> node) {
    std::string comm_group_name = node->pg_name<std::string>("");
    // [default communication group]
    // We assume that an empty comm group, or comm group '0' both correspond
    // to the default communicator group that includes all ranks.
    // If, in the future, we want to support a user-defined comm group '0',
    // revisit this logic.
    if (comm_group_name == "" || comm_group_name == "0") {
        // No communicator group is specified for this communication ET node.
        return nullptr;
    }

    int comm_group_id = std::stoi(comm_group_name);
    if (comm_groups.find(comm_group_id) == comm_groups.end()) {
        LoggerFactory::get_logger("workload")
            ->critical(
                "For rank {} ET node {}, communicator group {} not found",
                sys->id, node->id(), comm_group_id);
        exit(EXIT_FAILURE);
    }
    return comm_groups[comm_group_id];
}
