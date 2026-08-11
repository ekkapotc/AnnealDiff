#include <iostream>
#include <vector>
#include <thread>

#include "ADGraph.hpp"
#include "Optimizer.hpp"

int main() {
    ADGraph graph;
    current_graph = &graph;

    // Construct a complex expression graph to demonstrate search scalability
    ADVar x1(0.5), x2(1.2), x3(-0.8), x4(2.1);
    std::vector<int> input_ids = {x1.id, x2.id, x3.id, x4.id};

    ADVar h1 = sin(x1 * x2);
    ADVar h2 = cos(x2 * x3);
    ADVar h3 = tanh(x3 + x4);
    ADVar h4 = h1 * h2 + h3;
    ADVar h5 = sin(h2 + h3);
    ADVar h6 = cos(h4 * h5);

    ADVar y1 = h4 + h6;
    ADVar y2 = h5 * h6;
    std::vector<int> output_ids = {y1.id, y2.id};

    GraphAdjacency adj = extract_graph_adjacency(graph, input_ids, output_ids);

    std::cout << "=== Parallel Tempering SA for AD Vertex Elimination ===\n";
    std::cout << "Total Computational Graph Nodes: " << graph.next_id << "\n";
    std::cout << "Intermediate Nodes to Optimize:  " << adj.intermediates.size() << "\n\n";

    int num_threads = std::thread::hardware_concurrency();
    std::cout << "Running Parallel Tempering across " << num_threads << " concurrent threads...\n";

    ParallelTemperingResult pt_res = run_parallel_tempering_sa(
        adj, 
        /*num_replicas=*/ num_threads, 
        /*T_min=*/ 0.05, 
        /*T_max=*/ 50.0, 
        /*steps_per_sweep=*/ 40, 
        /*exchange_rounds=*/ 150
    );

    std::cout << "\n--- Optimization Results ---\n";
    std::cout << "Optimized Order:       ";
    for (int id : pt_res.best_order) std::cout << id << " ";
    std::cout << "\nMinimum Markowitz Cost: " << pt_res.min_cost << "\n";
    std::cout << "Total Replica Swaps:   " << pt_res.total_swaps_accepted << "\n";

    return 0;
}
