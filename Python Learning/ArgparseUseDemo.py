import argparse

parser = argparse.ArgumentParser(description="argparse demo")
parser.add_argument('--c',default=0,type=int)
opt = parser.parse_args()

a = opt.c

print(a)