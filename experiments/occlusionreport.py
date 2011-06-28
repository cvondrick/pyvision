import pickle
from pylab import *

loaded = pickle.load(open("occlusion.pkl"))

labeledframes = [0]

ranges = {
    "Visible": ([(270, 330), (469, 500)], "black"),
    "Partial Occlusion": ([(330, 375), (455, 469)], "green"),
    "Severe Occlusion": ([(390, 420)], "blue"),
    "Total Occlusion": ([(375, 390), (420, 455)], "red"),
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
#legend(loc = "upper left", numpoints=1)
ylim(0, 40)

show()
