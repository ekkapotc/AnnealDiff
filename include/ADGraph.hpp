#pragma once

#include <vector>
#include <utility>

// Global AD state declarations (defined in ADGraph.cpp)
class ADGraph;
extern thread_local ADGraph* current_graph;
extern thread_local int current_ad_depth;
extern thread_local bool s_grad_enabled;
extern const int MAX_AD_DEPTH;

// RAII Scope Guard to temporarily disable tape recording
class NoGradGuard {
private:
    bool prev_state;
public:
    NoGradGuard();
    ~NoGradGuard();
    
    // Prevent copying to enforce strict RAII semantics
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};

class ADVar;

// Fixed boundary snapshot of the tape
struct TapeSnapshot {
    int node_count;
    int size() const { return node_count; }
};

class ADGraph {
public:
    int next_id = 0;
    std::vector<double> node_values;
    std::vector<std::vector<std::pair<int, int>>> rev_adj;

public:
    int create_node(double val);
    void add_edge(int from, int to, int weight_node_id);
    
    // Returns a snapshot of the current graph boundary
    TapeSnapshot snapshot() const { return TapeSnapshot{ next_id }; }

    std::vector<ADVar> compute_gradient_graph(int output_id, const std::vector<int>& input_ids);
    std::vector<ADVar> compute_gradient_graph_custom_order(
        int output_id, 
        const std::vector<int>& input_ids, 
        const std::vector<int>& order
    );
};

class ADVar {
public:
    double val;
    int id;

    ADVar(double v = 0.0);
    ADVar(int existing_id, double v);

    friend ADVar operator+(const ADVar& l, const ADVar& r);
    friend ADVar operator-(const ADVar& l, const ADVar& r);
    friend ADVar operator*(const ADVar& l, const ADVar& r);
    friend ADVar operator/(const ADVar& l, const ADVar& r); 

    // Unary minus
    friend ADVar operator-(const ADVar& x);

    // Mixed Arithmetic (ADVar & double)
    friend ADVar operator+(const ADVar& l, double r);
    friend ADVar operator+(double l, const ADVar& r);
    friend ADVar operator-(const ADVar& l, double r);
    friend ADVar operator-(double l, const ADVar& r);
    friend ADVar operator*(const ADVar& l, double r);
    friend ADVar operator*(double l, const ADVar& r);
    friend ADVar operator/(const ADVar& l, double r);
    friend ADVar operator/(double l, const ADVar& r);

    // Transcendental functions
    // Unary minus
    friend ADVar operator-(const ADVar& x);

    // Mixed Arithmetic (ADVar & double)
    friend ADVar operator+(const ADVar& l, double r);
    friend ADVar operator+(double l, const ADVar& r);
    friend ADVar operator-(const ADVar& l, double r);
    friend ADVar operator-(double l, const ADVar& r);
    friend ADVar operator*(const ADVar& l, double r);
    friend ADVar operator*(double l, const ADVar& r);
    friend ADVar operator/(const ADVar& l, double r);
    friend ADVar operator/(double l, const ADVar& r);

    // Transcendental functions
    friend ADVar sin(const ADVar& x);
    friend ADVar cos(const ADVar& x);
    friend ADVar tan(const ADVar& x);
    friend ADVar exp(const ADVar& x);
    friend ADVar log(const ADVar& x);

    // Machine Learning Activations
    friend ADVar relu(const ADVar& x);
    friend ADVar sigmoid(const ADVar& x);
    friend ADVar tanh(const ADVar& x);

};

