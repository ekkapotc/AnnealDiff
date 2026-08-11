#include "Optimizer.hpp"
#include <random>
#include <algorithm>
#include <thread>
#include <cmath>

GraphAdjacency extract_graph_adjacency(
    const ADGraph& graph, 
    const std::vector<int>& input_ids, 
    const std::vector<int>& output_ids
) {
    GraphAdjacency adj;
    std::unordered_set<int> inputs(input_ids.begin(), input_ids.end());
    std::unordered_set<int> outputs(output_ids.begin(), output_ids.end());

    int n = graph.next_id;
    for (int v = 0; v < n; ++v) {
        for (const auto& edge : graph.rev_adj[v]) {
            int u = edge.first;
            if (u != v) {
                adj.in_edges[v].insert(u);
                adj.out_edges[u].insert(v);
            }
        }
    }

    for (int v = 0; v < n; ++v) {
        if (inputs.find(v) == inputs.end() && outputs.find(v) == outputs.end()) {
            if (!adj.in_edges[v].empty() || !adj.out_edges[v].empty()) {
                adj.intermediates.push_back(v);
            }
        }
    }
    return adj;
}

int evaluate_elimination_cost(
    std::unordered_map<int, std::unordered_set<int>> in_adj,
    std::unordered_map<int, std::unordered_set<int>> out_adj,
    const std::vector<int>& order
) {
    int total_cost = 0;

    for (int v : order) {
        int in_deg = static_cast<int>(in_adj[v].size());
        int out_deg = static_cast<int>(out_adj[v].size());

        total_cost += in_deg * out_deg;

        for (int u : in_adj[v]) {
            out_adj[u].erase(v);
            for (int w : out_adj[v]) {
                if (u != w) {
                    out_adj[u].insert(w);
                    in_adj[w].insert(u);
                }
            }
        }
        for (int w : out_adj[v]) {
            in_adj[w].erase(v);
        }

        in_adj[v].clear();
        out_adj[v].clear();
    }
    return total_cost;
}

struct Replica {
    int id;
    double temp;
    std::vector<int> order;
    int cost;
    std::mt19937 rng;

    Replica(int id, double temp, const std::vector<int>& init_order, unsigned int seed)
        : id(id), temp(temp), order(init_order), rng(seed) {}
};

void local_metropolis_sweep(Replica& rep, const GraphAdjacency& adj, int steps) {
    if (rep.order.size() < 2) return;

    std::uniform_int_distribution<size_t> dist(0, rep.order.size() - 1);
    std::uniform_real_distribution<double> rand_unif(0.0, 1.0);

    for (int step = 0; step < steps; ++step) {
        std::vector<int> neighbor_order = rep.order;
        size_t i = dist(rep.rng);
        size_t j = dist(rep.rng);
        while (i == j) j = dist(rep.rng);

        std::swap(neighbor_order[i], neighbor_order[j]);

        int neighbor_cost = evaluate_elimination_cost(adj.in_edges, adj.out_edges, neighbor_order);
        int delta_e = neighbor_cost - rep.cost;

        if (delta_e <= 0 || rand_unif(rep.rng) < std::exp(-static_cast<double>(delta_e) / rep.temp)) {
            rep.order = neighbor_order;
            rep.cost = neighbor_cost;
        }
    }
}

ParallelTemperingResult run_parallel_tempering_sa(
    const GraphAdjacency& adj,
    int num_replicas,
    double T_min,
    double T_max,
    int steps_per_sweep,
    int exchange_rounds
) {
    std::mt19937 master_rng(1337);
    std::uniform_real_distribution<double> rand_unif(0.0, 1.0);

    std::vector<Replica> replicas;
    for (int i = 0; i < num_replicas; ++i) {
        double T = T_min * std::pow(T_max / T_min, static_cast<double>(i) / (num_replicas - 1));
        std::vector<int> init_order = adj.intermediates;
        std::shuffle(init_order.begin(), init_order.end(), master_rng);

        int init_cost = evaluate_elimination_cost(adj.in_edges, adj.out_edges, init_order);
        replicas.emplace_back(i, T, init_order, master_rng());
        replicas.back().cost = init_cost;
    }

    std::vector<int> global_best_order = replicas[0].order;
    int global_best_cost = replicas[0].cost;
    int total_swaps = 0;

    for (int round = 0; round < exchange_rounds; ++round) {
        std::vector<std::thread> threads;
        threads.reserve(num_replicas);

        for (int i = 0; i < num_replicas; ++i) {
            threads.emplace_back(
                local_metropolis_sweep, 
                std::ref(replicas[i]), 
                std::cref(adj), 
                steps_per_sweep
            );
        }

        for (auto& t : threads) t.join();

        int start_idx = (round % 2 == 0) ? 0 : 1;
        for (int i = start_idx; i < num_replicas - 1; i += 2) {
            double beta_i = 1.0 / replicas[i].temp;
            double beta_next = 1.0 / replicas[i + 1].temp;
            double delta_pt = (beta_i - beta_next) * (replicas[i + 1].cost - replicas[i].cost);

            if (delta_pt <= 0.0 || rand_unif(master_rng) < std::exp(-delta_pt)) {
                std::swap(replicas[i].order, replicas[i + 1].order);
                std::swap(replicas[i].cost, replicas[i + 1].cost);
                total_swaps++;
            }
        }

        for (int i = 0; i < num_replicas; ++i) {
            if (replicas[i].cost < global_best_cost) {
                global_best_cost = replicas[i].cost;
                global_best_order = replicas[i].order;
            }
        }
    }

    return {global_best_order, global_best_cost, total_swaps};
}
