# compiler
CC = g++

# C++ compiler option
CXXFLAGS = -Wall -O2

# linker option
#LDFLAGS = -DDEBUG -g
LDFLAGS = 
LDFLAGS_INFO = -DJOIN_INFO

# source directory
SRC_DIR = ./src

# object directory
OBJ_DIR = ./obj
OBJ_DIR_INFO = ./obj/info

# exe file name
TARGET = main
TARGET_INFO = main_info

# source files to make
SRCS = $(notdir $(wildcard $(SRC_DIR)/*.cpp))

OBJS = $(SRCS:.cpp=.o)

OBJECTS = $(patsubst %.o,$(OBJ_DIR)/%.o,$(OBJS))
DEPS = $(OBJECTS:.o=.d)

OBJECTS_INFO = $(patsubst %.o,$(OBJ_DIR_INFO)/%.o,$(OBJS))
DEPS_INFO = $(OBJECTS_INFO:.o=.d)

all: $(TARGET)

info: $(TARGET_INFO)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@$(CC) $(CXXFLAGS) -c $< -o $@ -MD $(LDFLAGS)

$(OBJ_DIR_INFO)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@$(CC) $(CXXFLAGS) -c $< -o $@ -MD $(LDFLAGS_INFO)

$(TARGET) : $(OBJECTS)
	$(CC) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

$(TARGET_INFO) : $(OBJECTS_INFO)
	$(CC) $(CXXFLAGS) $(OBJECTS_INFO) -o $(TARGET_INFO) $(LDFLAGS_INFO)

.PHONY: clean all
clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGET) $(OBJECTS_INFO) $(DEPS_INFO) $(TARGET_INFO)

-include $(DEPS)
