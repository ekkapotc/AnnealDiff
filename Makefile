CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3 -pthread -Iinclude

SRC_DIR = src
INC_DIR = include
EX_DIR  = examples
BIN_DIR = bin
OBJ_DIR = obj

# 1. Core Library Objects
LIB_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
LIB_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/$(SRC_DIR)/%.o, $(LIB_SRCS))

# 2. Discover all test subdirectories in examples/
# Finds dirs like examples/test_case_01, examples/test_case_02
TEST_DIRS = $(wildcard $(EX_DIR)/*/)
# Strip trailing slashes and path to get binary names: test_case_01, test_case_02
TEST_NAMES = $(patsubst $(EX_DIR)/%/, %, $(TEST_DIRS))
TEST_BINS  = $(patsubst %, $(BIN_DIR)/%, $(TEST_NAMES))

# Default target builds all test executables
all: directories $(TEST_BINS)

# Dynamically generate build rules for each test directory
define TEST_RULE
$(BIN_DIR)/$(1): $$(patsubst $(EX_DIR)/$(1)/%.cpp, $(OBJ_DIR)/$(EX_DIR)/$(1)/%.o, $$(wildcard $(EX_DIR)/$(1)/*.cpp)) $(LIB_OBJS)
	@mkdir -p $$(BIN_DIR)
	$$(CXX) $$(CXXFLAGS) -o $$@ $$^
endef

$(foreach test,$(TEST_NAMES),$(eval $(call TEST_RULE,$(test))))

# Compile core library files
$(OBJ_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile example/test files in subdirectories
$(OBJ_DIR)/$(EX_DIR)/%.o: $(EX_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

directories:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)/$(SRC_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean directories
