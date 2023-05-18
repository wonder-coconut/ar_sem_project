import numpy as np
from matplotlib import pyplot as plt
import re
import sys

n = len(sys.argv)
if(n != 2):
    print("incorrect arguments")

filename = sys.argv[1]
n = int(filename[0]) #test data set size
f = open(filename,"r")
s = f.read().split("\n")
s.pop()
counts = []
images = []
frames = []

for i in range(0,n):
    counts.append([])
    for j in range(0,n):
        counts[i].append(0)

x = ".-+" #regex for dash line


i = -1
image = 0
for line in s:
    if(line[0] == '-'):
        i+=1
        image = int(re.sub(x,"",line))
        if(i >= n):
            i = images.index(image)
        else:
            images.append(image)
            frames.append(0)
    else:
        line = line.split()
        detection = int(line[2])
        counts[i][detection] += 1
        frames[i] += 1
        

print(images)
print(counts)
print(frames)
scaled_counts = []
for i in range(0,n):
    scaled_counts.append([])
    for count in counts[i]:
        count = count/frames[i]
        scaled_counts[i].append(count)

print(scaled_counts)