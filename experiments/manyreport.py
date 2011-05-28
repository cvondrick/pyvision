import pickle
from pylab import *

loaded = pickle.load(open("many.pkl"))

labeledframes = [0]

ranges = {
    "Visible": ([(0, 300), (1245, 1349)], "black"),
    "Occluded": ([(300, 404)], "red"),
    "Walking, No Jacket": ([(404, 584), (930, 985)], "blue"),
    "Crouching, No Jacket": ([(584, 687), (771, 985)], "green"),
    "Jumping, No Jacket": ([(687, 771)], "orange"),
    "Putting On Jacket": ([(985, 1245)], "purple")
}

for label, data in ranges.items():
    for start, stop in data[0]:
        use = dict(x for x in loaded.items() if start <= x[0] <= stop)
        use = use.items()
        use.sort()
        keys = [x[0] for x in use]
        values = [x[1] for x in use]
        plot(keys, values, color = data[1], linewidth=4, label = label)
        label = "_nolegend_"

diamondx, diamondy = max(((x[1], x[0]) for x in loaded.items()))
plot(diamondy, diamondx, "k*", label = "Requested Frame", markersize = 10)

xlabel("Frame")
ylabel("Expected Label Change")
#legend(numpoints=1)

show()
