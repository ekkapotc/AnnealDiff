#include <iostream>
#include <vector>
#include <algorithm>

#include "ADGraph.hpp"
#include "Optimizer.hpp"

int main() {
    std::cout << "==================================================\n";
    std::cout << "   NoGradGuard + SA Optimized Hessian Matrix      \n";
    std::cout << "==================================================\n";

    ADGraph graph;
    current_graph = &graph;

    // 1. Inputs
    ADVar x(2.0);
    ADVar y(3.0);
    std::vector<int> inputs = {x.id, y.id};

    // 2. Forward Pass: f(x, y) = x^3 + 2x^2y + y^2
    ADVar x2 = x * x;
    ADVar x3 = x2 * x;
    ADVar y2 = y * y;
    
    ADVar f = x3 + ADVar(2.0) * x2 * y + y2;

    std::cout << "f(x,y) = " << f.val << " (Expected: 41)\n";
    std::cout << "Graph size after forward pass: " << graph.snapshot().node_count << " nodes\n\n";

    // =========================================================
    // Phase 1: Inference Sweep using NoGradGuard
    // =========================================================
    std::cout << "--- Phase 1: Validation Sweep (NoGradGuard) ---\n";
    {
        NoGradGuard guard;
        for (int i = 0; i < 500; ++i) {
            ADVar val_x(2.0 + i * 0.01);
            ADVar val_y(3.0 - i * 0.01);
            ADVar val_f = (val_x * val_x * val_x) + ADVar(2.0) * (val_x * val_x) * val_y + (val_y * val_y);
        }
    }
    std::cout << "Validation pass completed (0 nodes added to tape).\n\n";

    // =========================================================
    // Phase 2: First-Order Derivatives (Gradient) with SA
    // =========================================================
    std::cout << "--- Phase 2: First Derivatives (Gradient) ---\n";
    
    // Extract adjacency for f(x, y)
    std::vector<int> out_f = {f.id};
    GraphAdjacency adj_grad = extract_graph_adjacency(graph, inputs, out_f);
    
    // Reverse mode baseline
    std::vector<int> rev_order_grad = adj_grad.intermediates;
    std::sort(rev_order_grad.rbegin(), rev_order_grad.rend());
    int rev_cost_grad = evaluate_elimination_cost(adj_grad.in_edges, adj_grad.out_edges, rev_order_grad);
    std::cout << "Reverse Mode Cost (Gradient) : " << rev_cost_grad << " FLOPs\n";
    
    // Run SA for the 1st derivative graph
    ParallelTemperingResult sa_grad = run_parallel_tempering_sa(adj_grad, 8, 0.1, 100.0, 30, 200);
    std::cout << "SA Optimized Cost (Gradient) : " << sa_grad.min_cost << " FLOPs\n";
    
    // Compute gradient using SA ordering
    std::vector<ADVar> grad_f = graph.compute_gradient_graph_custom_order(f.id, inputs, sa_grad.best_order);
    ADVar df_dx = grad_f[0];
    ADVar df_dy = grad_f[1];
    
    std::cout << "df/dx = " << df_dx.val << " (Expected: 36)\n";
    std::cout << "df/dy = " << df_dy.val << " (Expected: 14)\n";
    std::cout << "Graph size after 1st derivative: " << graph.snapshot().node_count << " nodes\n\n";

    // =========================================================
    // Phase 3: Second-Order Derivatives (Hessian) with SA
    // =========================================================
    std::cout << "--- Phase 3: Second Derivatives (Hessian) ---\n";
    
    // --- Hessian Row 1: d(df/dx) / d(x,y) ---
    std::vector<int> out_df_dx = {df_dx.id};
    GraphAdjacency adj_h1 = extract_graph_adjacency(graph, inputs, out_df_dx);
    
    std::vector<int> rev_order_h1 = adj_h1.intermediates;
    std::sort(rev_order_h1.rbegin(), rev_order_h1.rend());
    int rev_cost_h1 = evaluate_elimination_cost(adj_h1.in_edges, adj_h1.out_edges, rev_order_h1);
    
    ParallelTemperingResult sa_h1 = run_parallel_tempering_sa(adj_h1, 8, 0.1, 100.0, 30, 200);
    
    std::vector<ADVar> grad_df_dx = graph.compute_gradient_graph_custom_order(df_dx.id, inputs, sa_h1.best_order);
    
    std::cout << "[Hessian Row 1] Reverse Cost: " << rev_cost_h1 << " FLOPs\n";
    std::cout << "[Hessian Row 1] SA Cost     : " << sa_h1.min_cost << " FLOPs\n";
    std::cout << "d2f/dx2  = " << grad_df_dx[0].val << " (Expected: 24)\n";
    std::cout << "d2f/dxdy = " << grad_df_dx[1].val << " (Expected: 8)\n\n";
    
    // --- Hessian Row 2: d(df/dy) / d(x,y) ---
    std::vector<int> out_df_dy = {df_dy.id};
    GraphAdjacency adj_h2 = extract_graph_adjacency(graph, inputs, out_df_dy);
    
    std::vector<int> rev_order_h2 = adj_h2.intermediates;
    std::sort(rev_order_h2.rbegin(), rev_order_h2.rend());
    int rev_cost_h2 = evaluate_elimination_cost(adj_h2.in_edges, adj_h2.out_edges, rev_order_h2);
    
    ParallelTemperingResult sa_h2 = run_parallel_tempering_sa(adj_h2, 8, 0.1, 100.0, 30, 200);
    
    std::vector<ADVar> grad_df_dy = graph.compute_gradient_graph_custom_order(df_dy.id, inputs, sa_h2.best_order);
    
    std::cout << "[Hessian Row 2] Reverse Cost: " << rev_cost_h2 << " FLOPs\n";
    std::cout << "[Hessian Row 2] SA Cost     : " << sa_h2.min_cost << " FLOPs\n";
    std::cout << "d2f/dydx = " << grad_df_dy[0].val << " (Expected: 8)\n";
    std::cout << "d2f/dy2  = " << grad_df_dy[1].val << " (Expected: 2)\n";

    std::cout << "\nFinal Graph Size: " << graph.snapshot().node_count << " nodes\n";
    std::cout << "==================================================\n";

    return 0;
}
