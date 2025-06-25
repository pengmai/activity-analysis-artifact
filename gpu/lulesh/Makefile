# Environment Variables
CUDA_PATH ?= /usr/local/cuda-12.6
ENZYME_PATH ?= /home/jnix/Workspace/Enzyme/build/Enzyme/ClangEnzyme-20.so
LLENZYME_PATH ?= /home/jnix/Workspace/Enzyme/build/Enzyme/LLVMEnzyme-20.so
MLIR_TRANSLATE ?= mlir-translate
LLVM_BIN ?= /home/jnix/Workspace/llvm-project/build/bin
OPT ?= $(LLVM_BIN)/opt
EMLIR_OPT ?= /home/jnix/Workspace/Enzyme/build/Enzyme/MLIR/enzymemlir-opt
EMLIR_TRANSLATE ?= /home/jnix/Workspace/Enzyme/build/Enzyme/MLIR/enzymemlir-translate/enzymemlir-translate
CLANG_PATH ?= $(LLVM_BIN)/clang++
SM_VERSION  = 60

# Ablation testing & numerical verification
VERIFY  ?= no
FORWARD ?= yes
PHIOPT ?= no
BRANCHYOPT ?= no
MINCCACHE ?= yes
NEWCACHE ?= yes
OPTIMIZE ?= yes
RESTRICT ?= yes

SPECPHI ?= no
SELECT ?= no

DEBUG ?= no

CC := $(CLANG_PATH) 
CFLAGS  = -DNDEBUG -fno-exceptions -mllvm -enzyme-print -Rpass=enzyme \
 	-std=c++17 --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)\
	--no-cuda-version-check -Xclang -load -Xclang $(ENZYME_PATH)
# CFLAGS  = -DNDEBUG -fno-exceptions \
#  	-std=c++17 --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)\
# 	--no-cuda-version-check -Xclang
LDFLAGS = -lcudart_static -ldl -lrt -lpthread -lm 


ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3 
endif

ifeq ($(RESTRICT),yes)
	CFLAGS += -DRESTRICT=1
else
	CFLAGS += -DRESTRICT=0
endif

ifeq ($(DEBUG),yes)
	CFLAGS += -g
endif

ifeq ($(VERIFY),yes)
	CFLAGS += -DVERIFY
endif

# ifeq ($(MINCCACHE),yes)
# 	CFLAGS += -mllvm -enzyme-mincut-cache=1
# else
# 	CFLAGS += -mllvm -enzyme-mincut-cache=0
# endif

# ifeq ($(NEWCACHE),yes)
# 	CFLAGS += -mllvm -enzyme-new-cache=1
# else
# 	CFLAGS += -mllvm -enzyme-new-cache=0
# endif

# ifeq ($(SELECT),yes)
#   CFLAGS += -mllvm -enzyme-select-opt=1
# else
#   CFLAGS += -mllvm -enzyme-select-opt=0
# endif

# ifeq ($(SPECPHI),yes)
#   CFLAGS += -mllvm -enzyme-speculate-phis=1
# else
#   CFLAGS += -mllvm -enzyme-speculate-phis=0
# endif

ifeq ($(BRANCHYOPT),yes)
	CFLAGS += -DBRANCHYOPT
endif

# ifeq ($(PHIOPT),yes)
# 	CFLAGS += -mllvm -enzyme-phi-restructure=1
# else
# 	CFLAGS += -mllvm -enzyme-phi-restructure=0
# endif

ifeq ($(FORWARD),yes)
	CFLAGS += -DNormal_forward=1
else
	CFLAGS += -DNormal_forward=0
endif
#SILO_INCLUDES := /usr/local/silo-4.8/include
#SILO_LIBS := /usr/local/silo-4.8/lib

#LINKFLAGS = -lmpich -L$(MPICH_DIR)/lib 
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

all: release 

debug: LINKFLAGS += 

release: 	FLAGS += $(RFLAGS)
debug: 		FLAGS += $(DFLAGS)

release: lulesh
debug: lulesh

lulesh: allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o
	$(CC) $(CFLAGS) allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o -o lulesh $(LDFLAGS)

allocator.o: allocator.cu vector.h
	$(CC) $(CFLAGS) allocator.cu -I ./ -c -o allocator.o

lulesh.o: lulesh.cu util.h vector.h allocator.h
	$(CC) $(CFLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o lulesh.o

# lulesh.o: temp/lulesh_diff.ll
# 	$(CC) temp/lulesh_diff.ll -o lulesh.o --target=nvptx64-nvidia-cuda -march=sm_60

# lulesh_host.o: lulesh.cu temp/lulesh.fatbin
# 	$(CC) $(CFLAGS) lulesh.cu -I ./ $(INC_SILO) --cuda-host-only -relocatable-pch -Xclang -fcuda-include-gpubinary -Xclang temp/lulesh.fatbin -c -o lulesh_host.o

# lulesh.o: temp/lulesh_df_diff.ll
# 	$(CC) temp/lulesh_df_diff.ll -c -o lulesh.o --target=nvptx64-nvidia-cuda -march=sm_60

# temp/lulesh.ptx: temp/lulesh_df_diff.ll
# 	llc -march=nvptx64 -mcpu=sm_$(SM_VERSION) -mattr=+ptx64 temp/lulesh_df_diff.ll -o temp/lulesh.ptx

# temp/lulesh_df_diff.ll : temp/lulesh_annotated.ll
# 	$(OPT) --load-pass-plugin=$(LLENZYME_PATH) --passes="enzyme" temp/lulesh_annotated.ll -o temp/lulesh_df_diff.ll

# temp/lulesh_annotated.ll : temp/lulesh_annotated.mlir
# 	$(EMLIR_TRANSLATE) --activity-to-llvm temp/lulesh_annotated.mlir -o temp/lulesh_annotated.ll

# temp/lulesh_annotated.mlir : temp/lulesh.mlir
# 	$(EMLIR_OPT) temp/lulesh.mlir --print-activity-analysis="relative infer" -o temp/lulesh_annotated.mlir

# temp/lulesh.mlir: lulesh-cuda-nvptx64-nvidia-cuda-sm_60.bc
# 	$(EMLIR_TRANSLATE) lulesh-cuda-nvptx64-nvidia-cuda-sm_60.bc -import-llvm -o temp/lulesh.mlir

# Using Enzyme's default activity
# temp/lulesh_diff.ll: lulesh-cuda-nvptx64-nvidia-cuda-sm_60.bc
# 	$(OPT) --load-pass-plugin=$(LLENZYME_PATH) -passes="enzyme" lulesh-cuda-nvptx64-nvidia-cuda-sm_60.bc -o temp/lulesh_diff.ll

# lulesh-cuda-nvptx64-nvidia-cuda-sm_60.bc: lulesh.cu util.h vector.h allocator.h
# 	$(CC) $(CFLAGS) lulesh.cu -I ./ --cuda-device-only $(INC_SILO) -S -emit-llvm

lulesh-comms.o: lulesh-comms.cu
	$(CC) $(CFLAGS) lulesh-comms.cu -I ./ -c -o lulesh-comms.o

lulesh-comms-gpu.o: lulesh-comms-gpu.cu
	$(CC) $(CFLAGS) lulesh-comms-gpu.cu -I ./ -c -o lulesh-comms-gpu.o

clean: 
	rm -rf allocator.o lulesh-comms.o lulesh-comms-gpu.o lulesh.o lulesh xyz.asc regNumList.txt temp/*
