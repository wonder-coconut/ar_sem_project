import numpy as np
import matplotlib.pyplot as pyplot
import re
import sys

n = len(sys.argv)
if(n != 3):
    print("incorrect arguments")
    exit()

filename = sys.argv[1]
n = int(sys.argv[2])
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

        try:
            i = images.index(image)
        except(ValueError):
            images.append(image)
            frames.append(0)
    else:
        line = line.split()
        detection = int(line[2])
        counts[i][detection] += 1
        frames[i] += 1

scaled_counts = [] #scaling count data to frames
for i in range(0,n):
    scaled_counts.append([])
    for count in counts[i]:
        count = count/frames[i]
        scaled_counts[i].append(count)

bardata = []
imagelegend = []
i = j = 0
while(i < n):
    images[i] = "Image " + str(images[i])
    imagelegend.append(f"Image {i}")
    j = 0
    bardata.append([])
    while(j<n):
        bardata[i].append(scaled_counts[j][i])
        j+=1
    i+=1

ind = np.arange(n)
width = 0.07
i = 0
bars = []
colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan","k"]

pyplot.figure(figsize=(15,10.5))
while(i < n):
    bars.append(pyplot.bar(ind + width*i, bardata[i], width, color=colors[i]))
    i += 1

pyplot.xlabel("Images")
pyplot.ylabel("Detection count")
pyplot.title("Detection distribution")

pyplot.xticks(ind + width, images)
pyplot.legend(bars,imagelegend)
pyplot.show()