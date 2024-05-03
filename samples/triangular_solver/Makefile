#Holds the paths for library functions
lib_header_dir		:= ../../headers
lib_src_dir		:= ../../helper

#Get needed lib sources
lib_src_temp		:= mm2csr.c mmio.c sparsifymm2csr.c

# Function to prepend lib_src_dir to each src in the list
add_prefix = $(addprefix $(lib_src_dir)/,$1)

lib_src_list := $(foreach item,$(lib_src_temp),$(call add_prefix,$(item)))

#List of source files
CU_srcs			:= $(wildcard ./*.cu)

#List of objects
lib_objs		:= $(subst .c,.o,${lib_src_temp})
CU_objs			:= $(subst .cu,.o,${CU_srcs})
OBJS			:= $(lib_objs) $(CU_objs)

#Compilers
CC = gcc
NVCC = nvcc -ccbin g++
#flags
FLAGS = -O3 -I${lib_header_dir} -MMD -MP -lm -lcusparse

#**** Targets ****
cuTriSolve: ${OBJS}
	${NVCC} $^ -o $@ ${FLAGS}

#Pull in dependency info for existing .o files
-include $(OBJS:.o=.d)

#** Compile sources -> objects **
${lib_objs}: %.o: ${lib_src_dir}/%.c
	${CC} -c $< -o $@ ${FLAGS}

%.o: %.cu
	${NVCC} -dc ${FLAGS}  $< -o $@

.PHONY: clean help

help:
	@echo "** Makefile Targets and Description **"
	@echo "make - deafult option, builds the project"
	@echo "make clean - delete all object files and the project binary in current dir"

clean:
	rm -I -v ./cuTriSolve ./*.o ./*.d
