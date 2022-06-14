%module distributed_mcts


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

%include "std_string.i"

%include "kinarow.hpp"
%include "mcts.hpp"
 
  class NeuralNetwork
  {
  public:

    NeuralNetwork(std::string model_path, bool use_gpu, unsigned batch_size);
    void set_batch_size(unsigned batch_size) { this->batch_size_ = batch_size; }

    void reload_weights();

  };


%include <std_shared_ptr.i>
%shared_ptr(NeuralNetwork);