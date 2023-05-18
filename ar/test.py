import matplotlib.pyplot as pyplot
import numpy as np

n = 4

data = [[0.9021739130434783, 0.0, 0.09782608695652174, 0.0], [0.05806451612903226, 0.7354838709677419, 0.1935483870967742, 0.012903225806451613], [0.012345679012345678, 0.0, 0.9876543209876543, 0.0], [0.07462686567164178, 0.029850746268656716, 0.14925373134328357, 0.746268656716418]]

bardata = []
i = j = 0
while(i < n):
    j = 0
    bardata.append([])
    while(j<n):
        bardata[i].append(data[j][i])
        j+=1
    i+=1

ind = np.arange(n)
width = 0.25

pyplot.figure(figsize=(15,10.5))

bar1 = pyplot.bar(ind, bardata[0], width, color='r')
bar2 = pyplot.bar(ind + width,bardata[1], width, color='g')
bar3 = pyplot.bar(ind + width*2,bardata[2], width, color='b')
bar4 = pyplot.bar(ind + width*3,bardata[3], width, color='y')

pyplot.xlabel("Images")
pyplot.ylabel("Detection count")
pyplot.title("Detection distribution")

pyplot.xticks(ind + width, ["image 144","image 38", "image 56", "image 66"])
pyplot.legend((bar1,bar2,bar3,bar4),("Image 0","Image 1","Image 2","Image 3"))
pyplot.show()