import os
import sys

#import numpy as np
#import tensorflow as tf

def main():
    args = sys.argv
    userfile = ""

    if(len(args) > 1):
        userfile = args[1]
    else:
        userfile = "ERROR'\n'"

    print("python is processing file:", userfile)
    sys.stdout.flush() 

    with open("log/out.txt", 'w') as outfile:
        outfile.write(userfile)

if __name__ == "__main__":
    main()
