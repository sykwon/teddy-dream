# CC=g++
# CXXFLAGS=-Wall -O2 # C++ compiler option
# SRC_DIR=./src # source file directory
# OBJ_DIR=./obj # object file directory
# OBJS=main.o util.o
# TARGET=app.out # name of execution file


# $(TARGET): $(OBJS)
# 	$(CC) $(CXXFLAGS) -o $(TARGET)

# main.o : main.cpp util.h
# 	$(CC) $(CXXFLAGS) -c main.cpp

# util.o : util.h util.cpp
# 	$(CC) $(CXXFLAGS) -c util.cpp

# .PHONY: clean
# clean:
# 	rm -f *.o
# 	rm -f $(TARGET)
CC = g++

# C++ 컴파일러 옵션
CXXFLAGS = -Wall -O2

# 링커 옵션
#LDFLAGS = -DDEBUG -g
LDFLAGS = 
LDFLAGS_INFO = -DJOIN_INFO

# 소스 파일 디렉토리
SRC_DIR = ./src

# 오브젝트 파일 디렉토리
OBJ_DIR = ./obj
OBJ_DIR_INFO = ./obj/info

# 생성하고자 하는 실행 파일 이름
TARGET = main
TARGET_INFO = main_info

# Make 할 소스 파일들
# wildcard 로 SRC_DIR 에서 *.cpp 로 된 파일들 목록을 뽑아낸 뒤에
# notdir 로 파일 이름만 뽑아낸다.
# (e.g SRCS 는 foo.cpp bar.cpp main.cpp 가 된다.)
SRCS = $(notdir $(wildcard $(SRC_DIR)/*.cpp))

OBJS = $(SRCS:.cpp=.o)

# OBJS 안의 object 파일들 이름 앞에 $(OBJ_DIR)/ 을 붙인다.
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
