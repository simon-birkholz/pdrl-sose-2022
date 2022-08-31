%module(threads="1") distributed_mcts


%{
#define SWIG_FILE_WITH_INIT
#include "mcts.hpp"
#include "neural_network.hpp"
#include "kinarow.hpp"
#include <stdint.h>		// Use the C99 official header
#include <utility>
#include <string>
#include <stdexcept>
#include <cstring>
%}

%include <std_string.i>
%include <std_pair.i>
%include <std_vector.i>

%thread;
%include "kinarow.hpp"
%include "mcts.hpp"
%nothread;
  class NeuralNetwork
  {
  public:

    NeuralNetwork(std::string model_path, bool use_gpu, unsigned batch_size);
    void set_batch_size(unsigned batch_size) { this->batch_size_ = batch_size; }

    void unload_model();
    void reload_weights();

  };
  
%template(PII) std::pair<int,int>;
%template(IntVector) std::vector<int>;
%template(DoubleVector) std::vector<double>;

%template(SearchState) std::pair<std::vector<int>,int>;


%include <std_shared_ptr.i>
%shared_ptr(NeuralNetwork);