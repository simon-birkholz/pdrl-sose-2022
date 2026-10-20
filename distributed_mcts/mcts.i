%module(threads="1") distributed_mcts


%{
#define SWIG_FILE_WITH_INIT
#include "evaluator.hpp"
#include "kinarow.hpp"
#include "mcts.hpp"
#include "neural_network.hpp"
#include <stdexcept>
#include <string>
#include <utility>
%}

%include <exception.i>
%include <std_pair.i>
%include <std_string.i>
%include <std_vector.i>

%exception {
  try {
    $action
  } catch (const std::invalid_argument& e) {
    SWIG_exception(SWIG_ValueError, e.what());
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

// evaluate() returns a std::future, which python has no use for
%nodefaultctor Evaluator;
class Evaluator
{
public:
  virtual ~Evaluator();
};

class NeuralNetwork : public Evaluator
{
public:
  NeuralNetwork(std::string model_path, bool use_gpu, unsigned batch_size);
  void set_batch_size(unsigned batch_size);
  void reload_weights();
  void unload_model();
};

%template(PII) std::pair<int,int>;
%template(IntVector) std::vector<int>;
%template(DoubleVector) std::vector<double>;
%template(SearchState) std::pair<std::vector<int>,int>;

%thread;
%include "kinarow.hpp"
%include "mcts.hpp"
%nothread;
