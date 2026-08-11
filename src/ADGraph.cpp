#include "ADGraph.hpp"
#include <cmath>
#include <unordered_map>

// Global AD state definitions
thread_local ADGraph* current_graph = nullptr;
thread_local int current_ad_depth = 1;
thread_local bool s_grad_enabled = true;
const int MAX_AD_DEPTH = 3;

int ADGraph::create_node(double val) {
    int id = next_id++;
    node_values.push_back(val);
    rev_adj.push_back({});
    return id;
}

void ADGraph::add_edge(int from, int to, int weight_node_id) {
    rev_adj[to].push_back({from, weight_node_id});
}

std::vector<ADVar> ADGraph::compute_gradient_graph(int output_id, const std::vector<int>& input_ids) {
    std::unordered_map<int, ADVar> adjoints;
    adjoints.insert({output_id, ADVar(1.0)});

    for (int v = output_id; v >= 0; --v) {
        if (adjoints.find(v) != adjoints.end()) {
            ADVar grad_v = adjoints[v];
            for (auto& edge : rev_adj[v]) {
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

std::vector<ADVar> ADGraph::compute_gradient_graph_custom_order(
    int output_id, 
    const std::vector<int>& input_ids, 
    const std::vector<int>& order) 
{
    // Adjacency map: edges[u][v] represents the partial derivative d(v) / d(u)
    std::unordered_map<int, std::unordered_map<int, ADVar>> edges;
     
    // 1. Populate initial edges from the tape
    int original_node_count = next_id; // Capture size before generating new AD nodes
    for (int v = 0; v < original_node_count; ++v) {
        
        // Make a copy of the edges so reallocation of rev_adj doesn't crash us
        std::vector<std::pair<int, int>> current_edges = rev_adj[v]; 
        
        for (auto& edge : current_edges) {
            int u = edge.first;
            int weight_id = edge.second;
            ADVar new_weight(weight_id, node_values[weight_id]);
            
            // Accumulate parallel edges
            if (edges[u].count(v)) {
                edges[u][v] = edges[u][v] + new_weight;
            } else {
                edges[u][v] = new_weight;
            }
        }
    }

    // 2. Perform cross-country vertex elimination
    for (int v : order) {
        for (auto& in_edge : edges) {
            int u = in_edge.first;
            // If u points to the vertex 'v' being eliminated
            if (in_edge.second.count(v)) {
                ADVar du_v = in_edge.second[v];
                
                // Connect 'u' directly to all targets 'w' that 'v' points to
                for (auto& out_edge : edges[v]) {
                    int w = out_edge.first;
                    ADVar dv_w = out_edge.second;
                    ADVar msg = du_v * dv_w; // Chain rule
                    
                    if (edges[u].count(w)) {
                        edges[u][w] = edges[u][w] + msg;
                    } else {
                        edges[u][w] = msg;
                    }
                }
                in_edge.second.erase(v);
            }
        }
        edges.erase(v); // Remove the eliminated vertex completely
    }

    // 3. Extract final gradients for the requested inputs
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

ADVar::ADVar(double v) : val(v) {
    id = (current_graph && s_grad_enabled) ? current_graph->create_node(v) : -1;
}

ADVar::ADVar(int existing_id, double v) : val{v}, id{existing_id}{}

ADVar operator+(const ADVar& l, const ADVar& r) {
    ADVar res(l.val + r.val);
    if (current_graph && s_grad_enabled) {
        ADVar one(1.0);
        current_graph->add_edge(l.id, res.id, one.id);
        current_graph->add_edge(r.id, res.id, one.id);
    }
    return res;
}

ADVar operator-(const ADVar& l, const ADVar& r) {
    ADVar res(l.val - r.val);
    if (current_graph && s_grad_enabled) {
        ADVar one(1.0);
        ADVar neg_one(-1.0);
        current_graph->add_edge(l.id, res.id, one.id);
        current_graph->add_edge(r.id, res.id, neg_one.id);
    }
    return res;
}

ADVar operator*(const ADVar& l, const ADVar& r) {
    ADVar res(l.val * r.val);
    if (current_graph && s_grad_enabled) {
        current_graph->add_edge(l.id, res.id, r.id);
        current_graph->add_edge(r.id, res.id, l.id);
    }
    return res;
}

ADVar operator/(const ADVar& l, const ADVar& r) {
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
ADVar operator-(const ADVar& x) {
    return x * ADVar(-1.0);
}

// --- Mixed Arithmetic Operators ---
ADVar operator+(const ADVar& l, double r) { return l + ADVar(r); }
ADVar operator+(double l, const ADVar& r) { return ADVar(l) + r; }

ADVar operator-(const ADVar& l, double r) { return l - ADVar(r); }
ADVar operator-(double l, const ADVar& r) { return ADVar(l) - r; }

ADVar operator*(const ADVar& l, double r) { return l * ADVar(r); }
ADVar operator*(double l, const ADVar& r) { return ADVar(l) * r; }

ADVar operator/(const ADVar& l, double r) { return l / ADVar(r); }
ADVar operator/(double l, const ADVar& r) { return ADVar(l) / r; }

ADVar sin(const ADVar& x) {
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

ADVar cos(const ADVar& x) {
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

ADVar tan(const ADVar& x) {
    ADVar res(std::tan(x.val));
    if (current_graph && s_grad_enabled) {
       ADVar dx;
       if (current_ad_depth < MAX_AD_DEPTH) {
          current_ad_depth++;
          ADVar t = tan(x);
          dx = ADVar(1.0)+(t*t);
          current_ad_depth--;
       }else {
           double t_val = std::tan(x.val);
           dx = ADVar(1.0+(t_val*t_val));   
       }
       current_graph->add_edge(x.id, res.id, dx.id);
    }
    return res;
}

ADVar exp(const ADVar& x) {
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

ADVar log(const ADVar& x) {
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

ADVar tanh(const ADVar& x) {
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

ADVar relu(const ADVar& x) {
    ADVar res(std::max(0.0, x.val));
    if (current_graph && s_grad_enabled) {
        // Higher order derivatives of the step function are Dirac deltas, 
        // which are practically zero almost everywhere in standard ML contexts.
        ADVar dx = (x.val > 0.0) ? ADVar(1.0) : ADVar(0.0);
        current_graph->add_edge(x.id, res.id, dx.id);
    }
    return res;
}

ADVar sigmoid(const ADVar& x) {
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
