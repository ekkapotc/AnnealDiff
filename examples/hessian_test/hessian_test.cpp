#include <iostream>
#include <vector>
#include "ADGraph.hpp"

int main() {
    std::cout << "==================================================\n";
    std::cout << " NoGradGuard + Multivariate 2nd Order Derivatives \n";
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
    
    // Note: ADVar(2.0) introduces a constant node
    ADVar f = x3 + ADVar(2.0) * x2 * y + y2;

    std::cout << "f(x,y) = " << f.val << " (Expected: 41)\n";
    
    int nodes_after_f = graph.snapshot().node_count;
    std::cout << "Graph size after forward pass: " << nodes_after_f << " nodes\n\n";

    // =========================================================
    // Phase 1: Inference Sweep using NoGradGuard
    // =========================================================
    std::cout << "--- Phase 1: Validation Sweep (NoGradGuard) ---\n";
    {
        NoGradGuard guard;
        // Simulate a grid search or validation pass
        for (int i = 0; i < 500; ++i) {
            ADVar val_x(2.0 + i * 0.01);
            ADVar val_y(3.0 - i * 0.01);
            ADVar val_f = (val_x * val_x * val_x) + ADVar(2.0) * (val_x * val_x) * val_y + (val_y * val_y);
        }
    }
    
    int nodes_after_val = graph.snapshot().node_count;
    std::cout << "Nodes added during validation : " << (nodes_after_val - nodes_after_f) << "\n\n";

    // =========================================================
    // Phase 2: First-Order Derivatives
    // =========================================================
    std::cout << "--- Phase 2: First Derivatives (Gradient) ---\n";
    
    // Computing the gradient dynamically adds nodes to the tape to record the derivative math
    std::vector<ADVar> grad_f = graph.compute_gradient_graph(f.id, inputs);
    ADVar df_dx = grad_f[0];
    ADVar df_dy = grad_f[1];
    
    std::cout << "df/dx = " << df_dx.val << " (Expected: 36)\n";
    std::cout << "df/dy = " << df_dy.val << " (Expected: 14)\n";
    
    int nodes_after_grad = graph.snapshot().node_count;
    std::cout << "Graph size after 1st derivative: " << nodes_after_grad << " nodes\n\n";

    // =========================================================
    // Phase 3: Second-Order Derivatives (Hessian Matrix)
    // =========================================================
    std::cout << "--- Phase 3: Second Derivatives (Hessian) ---\n";
    
    // Hessian Row 1: Gradient of df/dx with respect to x and y
    std::vector<ADVar> grad_df_dx = graph.compute_gradient_graph(df_dx.id, inputs);
    std::cout << "d2f/dx2  = " << grad_df_dx[0].val << " (Expected: 24)\n";
    std::cout << "d2f/dxdy = " << grad_df_dx[1].val << " (Expected: 8)\n\n";
    
    // Hessian Row 2: Gradient of df/dy with respect to x and y
    std::vector<ADVar> grad_df_dy = graph.compute_gradient_graph(df_dy.id, inputs);
    std::cout << "d2f/dydx = " << grad_df_dy[0].val << " (Expected: 8)\n";
    std::cout << "d2f/dy2  = " << grad_df_dy[1].val << " (Expected: 2)\n";

    std::cout << "\nFinal Graph Size: " << graph.snapshot().node_count << " nodes\n";
    std::cout << "==================================================\n";

    return 0;
}
