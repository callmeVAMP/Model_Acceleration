# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/combined/home/parveen/varsha/sentiment_infer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/combined/home/parveen/varsha/sentiment_infer/build

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /mnt/combined/home/parveen/varsha/sentiment_infer/build/CMakeFiles /mnt/combined/home/parveen/varsha/sentiment_infer/build//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /mnt/combined/home/parveen/varsha/sentiment_infer/build/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named sentiment_infer

# Build rule for target.
sentiment_infer: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 sentiment_infer
.PHONY : sentiment_infer

# fast build rule for target.
sentiment_infer/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sentiment_infer.dir/build.make CMakeFiles/sentiment_infer.dir/build
.PHONY : sentiment_infer/fast

sentiment_infer.o: sentiment_infer.cpp.o
.PHONY : sentiment_infer.o

# target to build an object file
sentiment_infer.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sentiment_infer.dir/build.make CMakeFiles/sentiment_infer.dir/sentiment_infer.cpp.o
.PHONY : sentiment_infer.cpp.o

sentiment_infer.i: sentiment_infer.cpp.i
.PHONY : sentiment_infer.i

# target to preprocess a source file
sentiment_infer.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sentiment_infer.dir/build.make CMakeFiles/sentiment_infer.dir/sentiment_infer.cpp.i
.PHONY : sentiment_infer.cpp.i

sentiment_infer.s: sentiment_infer.cpp.s
.PHONY : sentiment_infer.s

# target to generate assembly for a file
sentiment_infer.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sentiment_infer.dir/build.make CMakeFiles/sentiment_infer.dir/sentiment_infer.cpp.s
.PHONY : sentiment_infer.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... sentiment_infer"
	@echo "... sentiment_infer.o"
	@echo "... sentiment_infer.i"
	@echo "... sentiment_infer.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

