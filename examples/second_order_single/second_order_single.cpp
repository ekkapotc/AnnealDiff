#include <iostream>
#include <vector>
#include <algorithm>

#include "ADGraph.hpp"
#include "Optimizer.hpp"

int main() {
    std::cout << "==================================================\n";
    std::cout << "  Higher-Order (2nd Derivative) Test with SA optimization \n";
    std::cout << "==================================================\n";

    // 1. Initialize Graph
    ADGraph graph;
    current_graph = &graph;

    // 2. Define Independent Variable
    ADVar x(3.0);
    std::vector<int> param_ids = {x.id};

    // 3. Build the Forward Graph: f(x) = x^3 + x^2
    ADVar x2 = x * x;
    ADVar x3 = x2 * x;
    ADVar y = x3 + x2;

    std::cout << "--- Forward Pass ---\n";
    std::cout << "f(x)   = " << y.val << " (Expected: 36)\n\n";

    // =========================================================
    // 4. Compute First Derivative (dy/dx) using SA
    // =========================================================
    std::cout << "--- First Derivative ---\n";
    GraphAdjacency adj1 = extract_graph_adjacency(graph, param_ids, {y.id});
    
    // Baseline Reverse-Mode Cost
    std::vector<int> reverse_order1 = adj1.intermediates;
    std::sort(reverse_order1.rbegin(), reverse_order1.rend()); 
    int reverse_cost1 = evaluate_elimination_cost(adj1.in_edges, adj1.out_edges, reverse_order1);

    // Optimized SA Cost
    ParallelTemperingResult sa_result1 = run_parallel_tempering_sa(
        adj1, 8, 0.1, 100.0, 30, 200
    );

    std::cout << "1st Deriv Reverse-Mode Cost : " << reverse_cost1 << " ops\n";
    std::cout << "1st Deriv SA Optimized Cost : " << sa_result1.min_cost << " ops\n";

    std::vector<ADVar> grads1 = graph.compute_gradient_graph_custom_order(y.id, param_ids, sa_result1.best_order);
    ADVar dy_dx = grads1[0];

    std::cout << "f'(x)  = " << dy_dx.val << " (Expected: 33)\n\n";

    // =========================================================
    // 5. Compute Second Derivative (d^2y/dx^2) using SA
    // =========================================================
    // Since dy_dx is an ADVar, we treat it as the new output node!
    std::cout << "--- Second Derivative ---\n";
    GraphAdjacency adj2 = extract_graph_adjacency(graph, param_ids, {dy_dx.id});
    
    // Baseline Reverse-Mode Cost (Notice the graph is larger now!)
    std::vector<int> reverse_order2 = adj2.intermediates;
    std::sort(reverse_order2.rbegin(), reverse_order2.rend());
    int reverse_cost2 = evaluate_elimination_cost(adj2.in_edges, adj2.out_edges, reverse_order2);

    // Optimized SA Cost
    ParallelTemperingResult sa_result2 = run_parallel_tempering_sa(
        adj2, 8, 0.1, 100.0, 30, 200
    );

    std::cout << "2nd Deriv Reverse-Mode Cost : " << reverse_cost2 << " ops\n";
    std::cout << "2nd Deriv SA Optimized Cost : " << sa_result2.min_cost << " ops\n";

    std::vector<ADVar> grads2 = graph.compute_gradient_graph_custom_order(dy_dx.id, param_ids, sa_result2.best_order);
    ADVar d2y_dx2 = grads2[0];

    std::cout << "f''(x) = " << d2y_dx2.val << " (Expected: 20)\n";
    std::cout << "==================================================\n";

    return 0;
}
