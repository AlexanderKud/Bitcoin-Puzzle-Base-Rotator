### This is a Modification of https://github.com/puzzleman22/bitcoin_puzzle_transformation

each thread has a base from 2 to 32 base

1) base random number generation
2) convert to base (depends on thread number)
3) rotate base
4) apply full transformations (inversion, reversion, shift left, shift up)
5) repeat with a new number until the target is found

# This Bitcoin Puzzle searcher Uses GPU with transformations

<img src="https://i.postimg.cc/G380GfD6/key-transformation-viz.png" />

## HOW TO USE:

main.exe 100000 1fffff 29a78213caa9eea824acf08022ab9dfc83414f56 - puzzle 21\
main.exe 1000000 1ffffff 2f396b29b27324300d0c59b17c3abc1835bd3dbb - puzzle 25\
main.exe 100000000 1ffffffff 4e15e5189752d1eaf444dfd6bff399feb0443977 - puzzle 33\
main.exe 10000000000 1ffffffffff d1562eb37357f9e6fc41cb2359f4d3eda4032329 - puzzle 41\
main.exe 1000000000000000000 1ffffffffffffffffff 105b7f253f0ebd7843adaebbd805c944bfb863e4 - puzzle 73\
main.exe 1000000000000000000000000000000000000000 1fffffffffffffffffffffffffffffffffffffff 242d790e5a168043c76f0539fd894b73ee67b3b3 - puzzle 157\
main.exe 10000000000000000000 1fffffffffffffffffff 783c138ac81f6a52398564bb17455576e8525b29 - puzzle 77

# NOTE

you set the amount of blocks and threads like this:

###  32 blocks x 32 threads = 1024 threads
main.exe 100000 1fffff 29a78213caa9eea824acf08022ab9dfc83414f56 32 32

# Compile with:

nvcc -o main main.cu

# USE ONLY THE PUZZLES THAT START WITH 1

## I'll explain how the transformation works:

1) a base random number is generated
2) it is converted into a binary like "110011"
3) a shift to the left is made (a full cycle)
4) at each shift of 1 bit, a vertical cycle is made 16 times in hex from 0 to F
5) a binary is reversed
6) bits are inverted
7) all this is done in cycles nested within each other

all these operations occur in thousands of threads, the more powerful the GPU, the more parallel transformations

all keys within one full transformation are unique, there are no repetitions.... unless the base number itself is repeated (which is unlikely on large ranges)

proof that it works, you can check it on short puzzles, it works like a clock, on large ranges it will take more time of course, but this is a chance to try your luck in a different style

this algorithm is available both on python (CPU), and on GPU

I believe this is an effective way to search

# for questions and other things
Author Telegram: https://t.me/nmn5436

if you liked this idea or found something, my BTC address for donations:
bc1p6fmhpep0wkqkzvw86fg0afz85fyw692vmcg05460yuqstva7qscs2d9xhk

# NOTE

the CPU version is here https://github.com/puzzleman22/Bitcoin-puzzle-transformations-CPU
