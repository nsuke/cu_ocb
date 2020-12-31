CUDA_ARCH=sm_61

NVCC=nvcc
NVCC_FLAGS=-I.. -arch=$(CUDA_ARCH) -std=c++17 -O2 --default-stream per-thread --compiler-options='-O2 -Wall -Wextra -fPIC -g'

PREFIX?=install

OBJ_DIR=.obj
BIN_DIR=bin
LIB_DIR=lib

OBJS=\
       $(OBJ_DIR)/camellia_cpu.o \
       $(OBJ_DIR)/camellia.o \
       $(OBJ_DIR)/checksum.o \
       $(OBJ_DIR)/offset.o \
       $(OBJ_DIR)/ocb_camellia.o \
       $(OBJ_DIR)/ocb_camellia_impl.o \

DEPS=$(OBJS:.o=.d)
DEPS += \
	$(OBJ_DIR)/main.d \
	$(OBJ_DIR)/test.d \

$(OBJ_DIR)/%.o : %.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -MP -MMD

$(OBJ_DIR)/%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -MP -MMD

STATIC=$(LIB_DIR)/libcu_ocb.a
$(STATIC): $(OBJS)
	ar rcs $(STATIC) $(OBJS)

SHARED=$(LIB_DIR)/libcu_ocb.so
$(SHARED): $(OBJS)
	$(NVCC) -o $(SHARED) $(OBJS) -shared

LINK_FLAGS=-L$(LIB_DIR) -lcu_ocb $(LDFLAGS)

EXE=$(BIN_DIR)/camellia
$(EXE): $(OBJ_DIR)/main.o $(STATIC) $(SHARED)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LINK_FLAGS)

TEST=$(BIN_DIR)/test
$(TEST): $(OBJ_DIR)/test.o $(STATIC) $(SHARED)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LINK_FLAGS)

all: $(EXE) $(STATIC) $(SHARED)
.DEFAULT_GOAL := all

check: $(TEST)
	$(TEST)

install: all ocb_camellia.h
	install -d $(DESTDIR)$(PREFIX)/bin/
	install -m 755 $(EXE) $(DESTDIR)$(PREFIX)/bin/
	install -d $(DESTDIR)$(PREFIX)/lib/
	install -m 644 $(STATIC) $(DESTDIR)$(PREFIX)/lib/
	install -m 644 $(SHARED) $(DESTDIR)$(PREFIX)/lib/
	install -d $(DESTDIR)$(PREFIX)/include/cu_ocb/
	install -m 644 ocb_camellia.h $(DESTDIR)$(PREFIX)/include/cu_ocb/

clean:
	$(RM) -r $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)

SIZE?=1k

/tmp/$(SIZE).blob:
	../../gendata.sh $(SIZE)

run: $(EXE) /tmp/$(SIZE).blob
	rm -f /tmp/$(SIZE).blob.gpg /tmp/$(SIZE).blob.decrypted
	LD_LIBRARY_PATH=$(LIB_DIR) time $(EXE) --data-size $(SIZE) && sha1sum /tmp/$(SIZE).blob*

-include $(DEPS)

$(info $(shell mkdir -p $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)))
