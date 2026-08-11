#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "ADGraph.hpp"
#include "Optimizer.hpp" 

// ---------------------------------------------------------
// Original 1D Heat Equation Simulation
// ---------------------------------------------------------
std::vector<double> simulate_heat_1d(double alpha_val, double T_left_val,
                                     int Nx, int Nt, double dx, double dt,
                                     std::vector<int>* param_ids_out = nullptr,
                                     ADVar* loss_out = nullptr,
                                     const std::vector<double>& target_u = {}) {
    ADVar alpha(alpha_val);
    ADVar T_left(T_left_val);
    if (param_ids_out) {
        param_ids_out->push_back(alpha.id);
        param_ids_out->push_back(T_left.id);
    }
    std::vector<ADVar> u(Nx, ADVar(20.0));
    u[0] = T_left;
    for (int n = 0; n < Nt; ++n) {
        std::vector<ADVar> unext = u;
        for (int i = 1; i < Nx - 1; ++i) {
            ADVar d2u_dx2 = (u[i + 1] - u[i] * 2.0 + u[i - 1]) / (dx * dx);
            unext[i] = u[i] + alpha * d2u_dx2 * dt;
        }
        unext[0] = T_left;
        unext[Nx - 1] = ADVar(20.0);
        u = unext;
    }
    if (loss_out && !target_u.empty()) {
        ADVar loss(0.0);
        for (int i = 0; i < Nx; ++i) {
            ADVar diff = u[i] - target_u[i];
            loss = loss + diff * diff * 0.5;
        }
        *loss_out = loss;
    }
    std::vector<double> u_vals(Nx);
    for (int i = 0; i < Nx; ++i) u_vals[i] = u[i].val;
    return u_vals;
}

int main() {
    int Nx = 5, Nt = 10;
    double L = 1.0, T_end = 0.1;
    double dx = L / (Nx - 1);
    double dt = T_end / Nt;
    double true_alpha = 0.015, true_T_left = 80.0;

    std::cout << "--- [Inverse Heat] Generating Synthetic Target Data ---\n";
    std::vector<double> target_profile;
    {
        ADGraph temp_graph;
        current_graph = &temp_graph;
        target_profile = simulate_heat_1d(true_alpha, true_T_left, Nx, Nt, dx, dt);
    }

    double estimated_alpha = 0.005, estimated_T_left = 50.0;
    int epochs = 200;
    double lr_alpha = 1e-6, lr_T = 0.1;

    // Cache for the optimized elimination order
    std::vector<int> optimized_order;
    bool order_optimized = false;

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        ADGraph local_graph;
        current_graph = &local_graph;
        std::vector<int> param_ids;
        ADVar loss(0.0);

        // 1. Build the graph for this epoch
        simulate_heat_1d(estimated_alpha, estimated_T_left, Nx, Nt, dx, dt,
                         &param_ids, &loss, target_profile);

        // 2. Run SA on the first epoch to find the optimal cross-country order
        if (!order_optimized) {
            std::cout << "Running Parallel Tempering SA to find optimal elimination order...\n";
            
            // Extract adjacency using your existing function
            std::vector<int> output_ids = {loss.id};
            GraphAdjacency adj = extract_graph_adjacency(local_graph, param_ids, output_ids);
            
            // Run your advanced SA optimizer
            ParallelTemperingResult sa_result = run_parallel_tempering_sa(
                adj, 8, 0.1, 100.0, 30, 200
            );
            
            optimized_order = sa_result.best_order;
            order_optimized = true;
            
            std::cout << "Optimization complete. Best cost: " << sa_result.min_cost 
                      << " | Swaps: " << sa_result.total_swaps_accepted << "\n";
        }

        // 3. Compute gradients using the optimized order
        std::vector<ADVar> grads = local_graph.compute_gradient_graph_custom_order(
            loss.id, param_ids, optimized_order
        );

        if (epoch % 50 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch
                      << " | Loss: " << std::fixed << std::setprecision(6) << loss.val
                      << " | alpha = " << estimated_alpha
                      << " | T_left = " << estimated_T_left << "\n";
        }
        
        // Update parameters
        estimated_alpha -= lr_alpha * grads[0].val;
        estimated_T_left -= lr_T * grads[1].val;
        if (estimated_alpha < 1e-5) estimated_alpha = 1e-5;
    }

    std::cout << "\n--- Final Results ---\n";
    std::cout << "Recovered alpha  : " << estimated_alpha << "\n";
    std::cout << "Recovered T_left : " << estimated_T_left << "\n";
    return 0;
}

