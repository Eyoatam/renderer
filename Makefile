CC := clang 
CFLAGS := -Xclang -fopenmp -Wall -g -O3
LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp
CFLAGS += -Ilib/stb -I/opt/homebrew/opt/libomp/include 

SRC := $(wildcard src/*.c)
OBJS := $(SRC:.c=.o)
BIN := bin

.PHONY: all clean

all: dirs renderer

dirs:
	mkdir -p ./$(BIN)

renderer: $(OBJS)
	$(CC) -o $(BIN)/renderer $^ $(LDFLAGS)

run: all
	$(BIN)/renderer

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS) 

OUT := $(wildcard *.png)

clean: 
	rm -rf $(BIN) $(OBJS) $(OUT)
