#include "ADGraph.hpp"

#include <cmath>
#include <set>
#include <unordered_map>
#include <iostream>

// Global AD state definitions
thread_local ADGraph *current_graph = nullptr;
thread_local int current_ad_depth = 1;
thread_local bool s_grad_enabled = true;
const int MAX_AD_DEPTH = 3;

// --- RAII Guard Implementation ---
NoGradGuard::NoGradGuard() : prev_state(s_grad_enabled) {
  s_grad_enabled = false;
}

NoGradGuard::~NoGradGuard() { s_grad_enabled = prev_state; }

int ADGraph::create_node(double val) {
  int id = next_id++;
  node_values.push_back(val);
  rev_adj.push_back({});
  return id;
}

void ADGraph::add_edge(int from, int to, int weight_node_id) {
  rev_adj[to].push_back({from, weight_node_id});
}

// --- Gradient Computation with TapeSnapshot ---
std::vector<ADVar>
ADGraph::compute_gradient_graph(int output_id,
                                const std::vector<int> &input_ids) {
  std::unordered_map<int, ADVar> adjoints;
  adjoints.insert({output_id, ADVar(1.0)});

  for (int v = output_id; v >= 0; --v) {
    if (adjoints.find(v) != adjoints.end()) {
      ADVar grad_v = adjoints[v];

      // Local snapshot of edges protects against iterator invalidation during
      // rev_adj reallocations
      std::vector<std::pair<int, int>> current_edges = rev_adj[v];

      for (auto &edge : current_edges) {
        int u = edge.first;
        ADVar weight_var(edge.second, node_values[edge.second]);
        ADVar msg = grad_v * weight_var;

        if (adjoints.find(u) == adjoints.end()) {
          adjoints.insert({u, msg});
        } else {
          adjoints[u] = adjoints[u] + msg;
        }
      }
    }
  }

  std::vector<ADVar> gradients;
  for (int id : input_ids) {
    gradients.push_back(adjoints.count(id) ? adjoints[id] : ADVar(0.0));
  }
  return gradients;
}

std::vector<ADVar>
ADGraph::compute_gradient_graph_custom_order(int output_id,
                                             const std::vector<int> &input_ids,
                                             const std::vector<int> &order) {
  std::unordered_map<int, std::unordered_map<int, ADVar>> edges;

  // Freeze the graph iteration boundary using TapeSnapshot
  TapeSnapshot tape = snapshot();

  // 1. Populate initial edges up to the snapshot limit
  for (int v = 0; v < tape.size(); ++v) {
    std::vector<std::pair<int, int>> current_edges = rev_adj[v];

    for (auto &edge : current_edges) {
      int u = edge.first;
      int weight_id = edge.second;
      ADVar new_weight(weight_id, node_values[weight_id]);

      if (edges[u].count(v)) {
        edges[u][v] = edges[u][v] + new_weight;
      } else {
        edges[u][v] = new_weight;
      }
    }
  }

  // 2. Cross-country elimination
  for (int v : order) {
    for (auto &in_edge : edges) {
      int u = in_edge.first;
      if (in_edge.second.count(v)) {
        ADVar du_v = in_edge.second[v];

        for (auto &out_edge : edges[v]) {
          int w = out_edge.first;
          ADVar dv_w = out_edge.second;
          ADVar msg = du_v * dv_w;

          if (edges[u].count(w)) {
            edges[u][w] = edges[u][w] + msg;
          } else {
            edges[u][w] = msg;
          }
        }
        in_edge.second.erase(v);
      }
    }
    edges.erase(v);
  }

  // 3. Extract gradients
  std::vector<ADVar> gradients;
  for (int id : input_ids) {
    if (edges.count(id) && edges[id].count(output_id)) {
      gradients.push_back(edges[id][output_id]);
    } else {
      gradients.push_back(ADVar(0.0));
    }
  }
  return gradients;
}

ADGraph::~ADGraph() {
  if (current_graph == this) {
    current_graph = nullptr;
  }
}

void ADGraph::emit_gradient_code(
    int output_id, 
    const std::vector<int>& input_ids, 
    const std::vector<int>& order, 
    std::ostream& out
) const {
    out << "// ========================================================\n";
    out << "// Auto-Generated Static Gradient Pass\n";
    out << "// ========================================================\n";
    out << "#include <vector>\n";
    out << "#include <map>\n\n";


    out << "extern \"C\" void compute_static_gradients(\n";
    out << "    const std::map<std::pair<int, int>, double>& init_edges,\n";
    out << "    std::map<int, double>& out_grads\n) {\n\n";

    // active_edges[u][v] tracks whether an edge currently exists in the emitted code scope
    std::unordered_map<int, std::unordered_map<int, bool>> active_edges;

    out << "    // --- 1. Load Initial Jacobians (Edge Weights) ---\n";
    for (int v = 0; v < next_id; ++v) {
        for (const auto& edge : rev_adj[v]) {
            int u = edge.first;
            
            // Only declare the variable if we haven't seen this edge yet
            if (!active_edges[u].count(v)) {
                active_edges[u][v] = true;
                out << "    double e_" << u << "_" << v 
                    << " = init_edges.at({" << u << ", " << v << "});\n";
            }
        }
    }
    out << "\n";
 
    out << "    // --- 2. Vertex Elimination (Optimized Order) ---\n";
    for (int v : order) {
        // Find all in-edges to vertex 'v'
        std::vector<int> in_nodes;
        for (const auto& pair : active_edges) {
            if (pair.second.count(v)) {
                in_nodes.push_back(pair.first);
            }
        }

        // Find all out-edges from vertex 'v'
        std::vector<int> out_nodes;
        if (active_edges.count(v)) {
            for (const auto& pair : active_edges[v]) {
                out_nodes.push_back(pair.first);
            }
        }

        // Cross-multiply to eliminate 'v'
        bool eliminated_something = false;
        for (int u : in_nodes) {
            for (int w : out_nodes) {
                eliminated_something = true;
                if (active_edges[u].count(w)) {
                    // Edge already exists, accumulate (+, *)
                    out << "    e_" << u << "_" << w << " += e_" << u << "_" << v 
                        << " * e_" << v << "_" << w << ";\n";
                } else {
                    // New edge created (Fill-in), declare it
                    out << "    double e_" << u << "_" << w << " = e_" << u << "_" << v 
                        << " * e_" << v << "_" << w << ";\n";
                    active_edges[u][w] = true;
                }
            }
            // Remove the incoming edge to the eliminated node
            active_edges[u].erase(v); 
        }
        
        // Remove all outgoing edges from the eliminated node
        active_edges.erase(v);
        
        if (eliminated_something) {
            out << "    // (Eliminated Node " << v << ")\n\n";
        }
    }

    out << "    // --- 3. Extract Target Gradients ---\n";
    for (int id : input_ids) {
        if (active_edges.count(id) && active_edges[id].count(output_id)) {
            out << "    out_grads[" << id << "] = e_" << id << "_" << output_id << ";\n";
        } else {
            out << "    out_grads[" << id << "] = 0.0;\n";
        }
    }
    out << "}\n";
}

void ADGraph::emit_gradient_bare_metal_code(
    int output_id, 
    const std::vector<int>& input_ids, 
    const std::vector<int>& order, 
    std::ostream& out
) const {
    out << "// ========================================================\n";
    out << "// Auto-Generated Bare-Metal Static Gradient Pass\n";
    out << "// ========================================================\n";
    out << "#include <cmath>\n\n";

    out << "extern \"C\" void compute_static_gradients_fast(\n";
    out << "    const double* __restrict__ init_edges,\n";
    out << "    double* __restrict__ out_grads\n) {\n\n";

    // 1. Assign flat sequential indices to every edge in the graph
    std::unordered_map<int, std::unordered_map<int, std::vector<int>>> edge_indices;
    int flat_edge_counter = 0;

    for (int v = 0; v < next_id; ++v) {
        for (const auto& edge : rev_adj[v]) {
            int u = edge.first;
            edge_indices[u][v].push_back(flat_edge_counter++);
        }
    }

    std::unordered_map<int, std::unordered_map<int, bool>> active_edges;

    out << "    // --- 1. Load Initial Jacobians (Contiguous Pointer Offsets) ---\n";
    for (int v = 0; v < next_id; ++v) {
        for (const auto& edge : rev_adj[v]) {
            int u = edge.first;
            if (!active_edges[u].count(v)) {
                active_edges[u][v] = true;
                
                // Sum parallel edge indices if multiple edges connect u to v
                const auto& indices = edge_indices[u][v];
                out << "    double e_" << u << "_" << v << " = ";
                for (size_t i = 0; i < indices.size(); ++i) {
                    out << "init_edges[" << indices[i] << "]" 
                        << (i + 1 < indices.size() ? " + " : ";\n");
                }
            }
        }
    }
    out << "\n";

    out << "    // --- 2. Vertex Elimination (Optimized Order) ---\n";
    for (int v : order) {
        std::vector<int> in_nodes;
        for (const auto& pair : active_edges) {
            if (pair.second.count(v)) in_nodes.push_back(pair.first);
        }
        std::vector<int> out_nodes;
        if (active_edges.count(v)) {
            for (const auto& pair : active_edges[v]) out_nodes.push_back(pair.first);
        }

        bool eliminated = false;
        for (int u : in_nodes) {
            for (int w : out_nodes) {
                eliminated = true;
                if (active_edges[u].count(w)) {
                    out << "    e_" << u << "_" << w << " += e_" << u << "_" << v 
                        << " * e_" << v << "_" << w << ";\n";
                } else {
                    out << "    double e_" << u << "_" << w << " = e_" << u << "_" << v 
                        << " * e_" << v << "_" << w << ";\n";
                    active_edges[u][w] = true;
                }
            }
            active_edges[u].erase(v);
        }
        active_edges.erase(v);
        if (eliminated) out << "    // (Eliminated Node " << v << ")\n\n";
    }

    out << "    // --- 3. Extract Target Gradients ---\n";
    for (size_t i = 0; i < input_ids.size(); ++i) {
        int id = input_ids[i];
        if (active_edges.count(id) && active_edges[id].count(output_id)) {
            out << "    out_grads[" << i << "] = e_" << id << "_" << output_id << ";\n";
        } else {
            out << "    out_grads[" << i << "] = 0.0;\n";
        }
    }
    out << "}\n";
}

ADVar::ADVar(double v) : val(v) {
  id = (current_graph && s_grad_enabled) ? current_graph->create_node(v) : -1;
}

ADVar::ADVar(int existing_id, double v) : val{v}, id{existing_id} {}

ADVar operator+(const ADVar &l, const ADVar &r) {
  ADVar res(l.val + r.val);
  if (current_graph && s_grad_enabled) {
    ADVar one(1.0);
    current_graph->add_edge(l.id, res.id, one.id);
    current_graph->add_edge(r.id, res.id, one.id);
  }
  return res;
}

ADVar operator-(const ADVar &l, const ADVar &r) {
  ADVar res(l.val - r.val);
  if (current_graph && s_grad_enabled) {
    ADVar one(1.0);
    ADVar neg_one(-1.0);
    current_graph->add_edge(l.id, res.id, one.id);
    current_graph->add_edge(r.id, res.id, neg_one.id);
  }
  return res;
}

ADVar operator*(const ADVar &l, const ADVar &r) {
  ADVar res(l.val * r.val);
  if (current_graph && s_grad_enabled) {
    current_graph->add_edge(l.id, res.id, r.id);
    current_graph->add_edge(r.id, res.id, l.id);
  }
  return res;
}

ADVar operator/(const ADVar &l, const ADVar &r) {
  ADVar res(l.val / r.val);
  if (current_graph && s_grad_enabled) {
    // Partial derivative with respect to l: 1 / r
    ADVar dl(1.0 / r.val);
    // Partial derivative with respect to r: -l / r^2
    ADVar dr(-l.val / (r.val * r.val));

    current_graph->add_edge(l.id, res.id, dl.id);
    current_graph->add_edge(r.id, res.id, dr.id);
  }
  return res;
}

// --- Unary Minus ---
ADVar operator-(const ADVar &x) { return x * ADVar(-1.0); }

// --- Mixed Arithmetic Operators ---
ADVar operator+(const ADVar &l, double r) { return l + ADVar(r); }
ADVar operator+(double l, const ADVar &r) { return ADVar(l) + r; }

ADVar operator-(const ADVar &l, double r) { return l - ADVar(r); }
ADVar operator-(double l, const ADVar &r) { return ADVar(l) - r; }

ADVar operator*(const ADVar &l, double r) { return l * ADVar(r); }
ADVar operator*(double l, const ADVar &r) { return ADVar(l) * r; }

ADVar operator/(const ADVar &l, double r) { return l / ADVar(r); }
ADVar operator/(double l, const ADVar &r) { return ADVar(l) / r; }

ADVar sin(const ADVar &x) {
  ADVar res(std::sin(x.val));
  if (current_graph && s_grad_enabled) {
    ADVar dx;
    if (current_ad_depth < MAX_AD_DEPTH) {
      current_ad_depth++;
      dx = cos(x);
      current_ad_depth--;
    } else {
      dx = ADVar(std::cos(x.val));
    }
    current_graph->add_edge(x.id, res.id, dx.id);
  }
  return res;
}

ADVar cos(const ADVar &x) {
  ADVar res(std::cos(x.val));
  if (current_graph && s_grad_enabled) {
    ADVar dx;
    if (current_ad_depth < MAX_AD_DEPTH) {
      current_ad_depth++;
      dx = ADVar(-1.0) * sin(x);
      current_ad_depth--;
    } else {
      dx = ADVar(-std::sin(x.val));
    }
    current_graph->add_edge(x.id, res.id, dx.id);
  }
  return res;
}

ADVar tan(const ADVar &x) {
  ADVar res(std::tan(x.val));
  if (current_graph && s_grad_enabled) {
    ADVar dx;
    if (current_ad_depth < MAX_AD_DEPTH) {
      current_ad_depth++;
      ADVar t = tan(x);
      dx = ADVar(1.0) + (t * t);
      current_ad_depth--;
    } else {
      double t_val = std::tan(x.val);
      dx = ADVar(1.0 + (t_val * t_val));
    }
    current_graph->add_edge(x.id, res.id, dx.id);
  }
  return res;
}

ADVar exp(const ADVar &x) {
  ADVar res(std::exp(x.val));
  if (current_graph && s_grad_enabled) {
    ADVar dx;
    if (current_ad_depth < MAX_AD_DEPTH) {
      current_ad_depth++;
      dx = exp(x); // Derivative of e^x is e^x
      current_ad_depth--;
    } else {
      dx = ADVar(std::exp(x.val));
    }
    current_graph->add_edge(x.id, res.id, dx.id);
  }
  return res;
}

ADVar log(const ADVar &x) {
  ADVar res(std::log(x.val));
  if (current_graph && s_grad_enabled) {
    ADVar dx;
    if (current_ad_depth < MAX_AD_DEPTH) {
      current_ad_depth++;
      dx = ADVar(1.0) / x; // Derivative of ln(x) is 1/x
      current_ad_depth--;
    } else {
      dx = ADVar(1.0 / x.val);
    }
    current_graph->add_edge(x.id, res.id, dx.id);
  }
  return res;
}

// --- Machine Learning Activations ---

ADVar tanh(const ADVar &x) {
  ADVar res(std::tanh(x.val));
  if (current_graph && s_grad_enabled) {
    ADVar dx;
    if (current_ad_depth < MAX_AD_DEPTH) {
      current_ad_depth++;
      ADVar t = tanh(x);
      dx = ADVar(1.0) - (t * t);
      current_ad_depth--;
    } else {
      double t_val = std::tanh(x.val);
      dx = ADVar(1.0 - t_val * t_val);
    }
    current_graph->add_edge(x.id, res.id, dx.id);
  }
  return res;
}

ADVar relu(const ADVar &x) {
  ADVar res(std::max(0.0, x.val));
  if (current_graph && s_grad_enabled) {
    // Higher order derivatives of the step function are Dirac deltas,
    // which are practically zero almost everywhere in standard ML contexts.
    ADVar dx = (x.val > 0.0) ? ADVar(1.0) : ADVar(0.0);
    current_graph->add_edge(x.id, res.id, dx.id);
  }
  return res;
}

ADVar sigmoid(const ADVar &x) {
  double s_val = 1.0 / (1.0 + std::exp(-x.val));
  ADVar res(s_val);
  if (current_graph && s_grad_enabled) {
    ADVar dx;
    if (current_ad_depth < MAX_AD_DEPTH) {
      current_ad_depth++;
      ADVar s = sigmoid(x);
      dx = s * (ADVar(1.0) - s); // Chain rule for sigmoid
      current_ad_depth--;
    } else {
      dx = ADVar(s_val * (1.0 - s_val));
    }
    current_graph->add_edge(x.id, res.id, dx.id);
  }
  return res;
}
