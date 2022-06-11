%module zero

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

