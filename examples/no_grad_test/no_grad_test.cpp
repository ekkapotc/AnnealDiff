#include <iostream>
#include <vector>
#include <algorithm>

#include "ADGraph.hpp"
#include "Optimizer.hpp"

int main() {
    std::cout << "==================================================\n";
    std::cout << "      Combined NoGradGuard + SA Optimizer Test    \n";
    std::cout << "==================================================\n";

    // 1. Initialize AD Graph
    ADGraph graph;
    current_graph = &graph;

    // 2. Define Model Parameters & Compute Primary Forward Pass
    ADVar x1(1.5), x2(2.5);
    std::vector<int> param_ids = {x1.id, x2.id};

    // Primary forward graph structure
    ADVar v1 = x1 * x2;
    ADVar v2 = sin(v1);
    ADVar y1 = exp(v2);
    ADVar y2 = v1 * v2;
    std::vector<int> output_ids = {y1.id, y2.id};

    int base_nodes = graph.snapshot().node_count;
    std::cout << "Base Graph Tape Size : " << base_nodes << " nodes\n\n";

    // =========================================================
    // Phase 1: Inference & Evaluation Sweep using NoGradGuard
    // =========================================================
    std::cout << "--- 1. Validation Sweep (NoGradGuard Active) ---\n";
    
    {
        NoGradGuard guard; // Freezes AD tape recording
        
        // Simulate 500 validation forward passes across candidate values
        for (int i = 0; i < 500; ++i) {
            ADVar test_x1(1.0 + i * 0.01);
            ADVar test_x2(2.0 - i * 0.005);
            
            ADVar test_v1 = test_x1 * test_x2;
            ADVar test_v2 = sin(test_v1);
            ADVar test_y1 = exp(test_v2);
            ADVar test_y2 = test_v1 * test_v2;
        }
    } // NoGradGuard destroyed here -> gradient tracking automatically resumes

    int post_val_nodes = graph.snapshot().node_count;
    std::cout << "Nodes added during validation : " << (post_val_nodes - base_nodes) << " nodes\n";
    std::cout << "Tape Size post-validation    : " << post_val_nodes << " nodes\n\n";

    // =========================================================
    // Phase 2: Graph Extraction & SA Optimization
    // =========================================================
    std::cout << "--- 2. SA Elimination Order Search ---\n";
    
    // Extract sparse adjacency from the clean, unpolluted graph
    GraphAdjacency adj = extract_graph_adjacency(graph, param_ids, output_ids);

    // Compute baseline costs
    std::vector<int> forward_order = adj.intermediates;
    std::sort(forward_order.begin(), forward_order.end()); 
    int forward_cost = evaluate_elimination_cost(adj.in_edges, adj.out_edges, forward_order);

    std::vector<int> reverse_order = adj.intermediates;
    std::sort(reverse_order.rbegin(), reverse_order.rend());
    int reverse_cost = evaluate_elimination_cost(adj.in_edges, adj.out_edges, reverse_order);

    // Run Simulated Annealing optimizer
    ParallelTemperingResult sa_result = run_parallel_tempering_sa(
        adj, 8, 0.1, 100.0, 30, 200
    );

    std::cout << "Forward Mode Cost  : " << forward_cost << " FLOPs\n";
    std::cout << "Reverse Mode Cost  : " << reverse_cost << " FLOPs\n";
    std::cout << "SA Optimized Cost  : " << sa_result.min_cost << " FLOPs\n\n";

    // =========================================================
    // Phase 3: Apply SA Best Ordering to Compute Gradients
    // =========================================================
    std::cout << "--- 3. Gradient Calculation with SA Order ---\n";

    for (size_t i = 0; i < output_ids.size(); ++i) {
        int out_id = output_ids[i];
        
        // Compute gradients using the SA optimal cross-country vertex sequence
        std::vector<ADVar> grads = graph.compute_gradient_graph_custom_order(
            out_id, param_ids, sa_result.best_order
        );

        std::cout << "Output Node (ID " << out_id << "): "
                  << "dy/dx1 = " << grads[0].val << ", "
                  << "dy/dx2 = " << grads[1].val << "\n";
    }

    std::cout << "\nFinal Graph Tape Size: " << graph.snapshot().node_count << " nodes\n";
    std::cout << "==================================================\n";

    return 0;
}
