#pragma once

#include "ADGraph.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct GraphAdjacency {
  std::unordered_map<int, std::unordered_set<int>> in_edges;
  std::unordered_map<int, std::unordered_set<int>> out_edges;
  std::vector<int> intermediates;
};

struct ParallelTemperingResult {
  std::vector<int> best_order;
  int min_cost;
  int total_swaps_accepted;
};

int evaluate_elimination_cost(
    std::unordered_map<int, std::unordered_set<int>> in_adj,
    std::unordered_map<int, std::unordered_set<int>> out_adj,
    const std::vector<int> &order);

GraphAdjacency extract_graph_adjacency(const ADGraph &graph,
                                       const std::vector<int> &input_ids,
                                       const std::vector<int> &output_ids);

ParallelTemperingResult
run_parallel_tempering_sa(const GraphAdjacency &adj, int num_replicas = 8,
                          double T_min = 0.1, double T_max = 100.0,
                          int steps_per_sweep = 30, int exchange_rounds = 200);

ParallelTemperingResult run_parallel_tempering_sa_seeded(
    const GraphAdjacency &adj, const std::vector<int> &seed_order,
    int num_replicas = 8, double min_temp = 0.1, double max_temp = 100.0,
    int num_steps = 30, int steps_per_temp = 200);
