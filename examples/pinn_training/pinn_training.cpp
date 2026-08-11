#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

#include "ADGraph.hpp"
#include "Optimizer.hpp"

// Lightweight Adam Optimizer
class AdamOptimizer {
private:
    double alpha, beta1, beta2, epsilon;
    int t;
    std::vector<double> m, v;

public:
    AdamOptimizer(int num_params, double lr = 0.008, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : alpha(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), m(num_params, 0.0), v(num_params, 0.0) {}

    void update(std::vector<double>& weights, const std::vector<ADVar>& grads) {
        t++;
        double bias_corr1 = 1.0 - std::pow(beta1, t);
        double bias_corr2 = 1.0 - std::pow(beta2, t);

        for (size_t i = 0; i < weights.size(); ++i) {
            double g = grads[i].val;
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g);
            weights[i] -= alpha * (m[i] / bias_corr1) / (std::sqrt(v[i] / bias_corr2) + epsilon);
        }
    }
};

int main() {
    std::cout << "==================================================\n";
    std::cout << "  MLP PINN Training (Adam + SA Optimized Graph)   \n";
    std::cout << "==================================================\n";

    // 1. Hyperparameters & Initialization
    double lr = 0.008;
    int epochs = 3000;
    int hidden_dim = 10;
    int num_params = hidden_dim * 3 + 1;
    std::vector<double> t_points = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};

    std::vector<double> weights(num_params);
    for(int i = 0; i < num_params; ++i) weights[i] = 0.1 * ((i % 11) - 5.0); 
    AdamOptimizer adam(num_params, lr);

    // Helper Lambda: Builds the exact same graph deterministically
    auto build_pinn_graph = [&](ADGraph& graph, std::vector<ADVar>& ad_params) -> ADVar {
        auto mlp_forward = [&](ADVar t) {
            ADVar out(0.0);
            for(int i = 0; i < hidden_dim; ++i) {
                out = out + ad_params[2 * hidden_dim + i] * 
                      tanh(ad_params[i] * t + ad_params[hidden_dim + i]);
            }
            return out + ad_params[3 * hidden_dim];
        };

        // IC Loss
        ADVar t_ic(0.0);
        ADVar x_ic = mlp_forward(t_ic);
        std::vector<ADVar> grad_ic = graph.compute_gradient_graph(x_ic.id, {t_ic.id});
        ADVar ic_loss = (x_ic - ADVar(1.0)) * (x_ic - ADVar(1.0)) + 
                        (grad_ic[0] - ADVar(0.0)) * (grad_ic[0] - ADVar(0.0));

        // PDE Loss
        ADVar pde_loss(0.0);
        for (double t_val : t_points) {
            ADVar t(t_val);
            ADVar x_pred = mlp_forward(t);
            std::vector<ADVar> grad_x = graph.compute_gradient_graph(x_pred.id, {t.id});
            std::vector<ADVar> grad_dx = graph.compute_gradient_graph(grad_x[0].id, {t.id});
            pde_loss = pde_loss + ((grad_dx[0] + x_pred) * (grad_dx[0] + x_pred));
        }
        return ic_loss + pde_loss * ADVar(1.0 / t_points.size());
    };

    // =========================================================
    // Phase 1: The "Dry Run" - Extract Topology & Run SA
    // =========================================================
    std::cout << "--- Phase 1: SA Topology Optimization ---\n";
    std::cout << "Extracting 17,000+ node graph. Running Parallel Tempering SA...\n";
    
    std::vector<int> optimal_order;
    {
        ADGraph dry_run_graph;
        current_graph = &dry_run_graph;
        
        std::vector<ADVar> dry_params;
        std::vector<int> param_ids;
        for(int i = 0; i < num_params; ++i) {
            dry_params.push_back(ADVar(weights[i]));
            param_ids.push_back(dry_params.back().id);
        }

        ADVar total_loss = build_pinn_graph(dry_run_graph, dry_params);
        
        GraphAdjacency adj = extract_graph_adjacency(dry_run_graph, param_ids, {total_loss.id});
        
        // Reverse mode baseline cost
        std::vector<int> rev_order = adj.intermediates;
        std::sort(rev_order.rbegin(), rev_order.rend());
        int rev_cost = evaluate_elimination_cost(adj.in_edges, adj.out_edges, rev_order);
        
        // Because the graph is massive, we use slightly fewer SA iterations to keep it fast
        ParallelTemperingResult sa_res = run_parallel_tempering_sa(adj, 8, 0.1, 100.0, 15, 100);
        optimal_order = sa_res.best_order;

        std::cout << "Reverse Mode FLOPs : " << rev_cost << "\n";
        std::cout << "SA Optimized FLOPs : " << sa_res.min_cost << "\n";
        std::cout << "FLOPs saved per epoch : " << (rev_cost - sa_res.min_cost) << "\n\n";
    }

    // =========================================================
    // Phase 2: High-Speed Training Loop using Cached SA Order
    // =========================================================
    std::cout << "--- Phase 2: Training (3000 Epochs) ---\n";
    
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        ADGraph graph;
        current_graph = &graph;

        std::vector<ADVar> ad_params;
        std::vector<int> param_ids;
        for(int i = 0; i < num_params; ++i) {
            ad_params.push_back(ADVar(weights[i]));
            param_ids.push_back(ad_params.back().id);
        }

        ADVar total_loss = build_pinn_graph(graph, ad_params);

        // USE THE SA OPTIMIZED ORDER HERE!
        std::vector<ADVar> weight_grads = graph.compute_gradient_graph_custom_order(
            total_loss.id, param_ids, optimal_order
        );

        adam.update(weights, weight_grads);

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << std::setw(4) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(5) << total_loss.val 
                      << " | Tape Nodes: " << graph.snapshot().node_count << "\n";
        }
    }

    // ====================================================
    // Phase 3: Final Evaluation
    // ====================================================
    std::cout << "\n--- Final Network Evaluation ---\n";
    std::cout << "Time(t)\t | Network x(t)\t | Target cos(t)\n";
    std::cout << "----------------------------------------\n";
    
    NoGradGuard guard;
    for (double t_val : t_points) {
        ADVar t(t_val);
        std::vector<ADVar> eval_params;
        for(double w : weights) eval_params.push_back(ADVar(w));

        ADVar out(0.0);
        for(int i = 0; i < hidden_dim; ++i) {
            out = out + eval_params[2 * hidden_dim + i] * 
                  tanh(eval_params[i] * t + eval_params[hidden_dim + i]);
        }
        ADVar x_pred = out + eval_params[3 * hidden_dim];
        
        std::cout << std::fixed << std::setprecision(1) << t_val << "\t | " 
                  << std::setprecision(4) << x_pred.val << "\t | " 
                  << cos(t_val) << "\n";
    }
    std::cout << "==================================================\n";

    return 0;
}
