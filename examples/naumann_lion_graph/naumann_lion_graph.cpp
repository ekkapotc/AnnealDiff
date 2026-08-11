#include <iostream>
#include <vector>
#include <algorithm>

#include "ADGraph.hpp"
#include "Optimizer.hpp"

int main() {
    std::cout << "==================================================\n";
    std::cout << "          Naumann's Lion Graph Test Case          \n";
    std::cout << "==================================================\n";

    ADGraph graph;
    current_graph = &graph;

    // 1. Independent Variables (Inputs: -1, 0)
    ADVar v_minus_1(1.5); 
    ADVar v_0(2.5);
    std::vector<int> param_ids = {v_minus_1.id, v_0.id};

    // 2. Intermediate Variables (1, 2)
    // v_1 depends on v_{-1} and v_0
    ADVar v_1 = v_minus_1 * v_0;
    
    // v_2 depends only on v_1
    ADVar v_2 = sin(v_1);

    // 3. Dependent Variables (Outputs: 3, 4, 5, 6)
    // v_3, v_4, v_5 depend only on v_2
    ADVar v_3 = exp(v_2);
    ADVar v_4 = cos(v_2);
    ADVar v_5 = tan(v_2);
    
    // v_6 depends on both v_1 and v_2
    ADVar v_6 = v_1 * v_2;
    
    std::vector<int> output_ids = {v_3.id, v_4.id, v_5.id, v_6.id};

    // Extract adjacency for the optimizer
    GraphAdjacency adj = extract_graph_adjacency(graph, param_ids, output_ids);

    // --- Baseline 1: Forward-Mode (Topological) ---
    std::vector<int> forward_order = adj.intermediates;
    std::sort(forward_order.begin(), forward_order.end()); 
    int forward_cost = evaluate_elimination_cost(adj.in_edges, adj.out_edges, forward_order);

    // --- Baseline 2: Reverse-Mode (Reverse Topological) ---
    std::vector<int> reverse_order = adj.intermediates;
    std::sort(reverse_order.rbegin(), reverse_order.rend());
    int reverse_cost = evaluate_elimination_cost(adj.in_edges, adj.out_edges, reverse_order);

    // --- Advanced Optimizer: Parallel Tempering SA ---
    std::cout << "Running Parallel Tempering SA optimizer...\n";
    ParallelTemperingResult sa_result = run_parallel_tempering_sa(
        adj, 8, 0.1, 100.0, 30, 200
    );

    // --- Print Results ---
    std::cout << "\n--- Elimination Cost Comparison ---\n";
    std::cout << "Strict Forward-Mode Cost : " << forward_cost << " ops\n";
    std::cout << "Strict Reverse-Mode Cost : " << reverse_cost << " ops\n";
    std::cout << "Optimal Cross-Country    : " << sa_result.min_cost << " ops\n";
    std::cout << "-----------------------------------\n";
    
    std::cout << "Optimal Vertex Sequence  : [";
    for (size_t i = 0; i < sa_result.best_order.size(); ++i) {
        std::cout << sa_result.best_order[i];
        if (i != sa_result.best_order.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;
}
