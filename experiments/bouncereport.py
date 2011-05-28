import pickle
from pylab import *

loaded = pickle.load(open("bounce.pkl"))

ranges = {
    "Wrong Box": ([

                     (40, 60),
                     (80, 100),
                     (120, 140)], "orange"),
    "Correct Box": ([(0, 20), (20, 40),
                   (60, 80),
                   (100, 120),
                   (140, 160)], "green"),
#    "Not a Person": ([(950, 1052), (1267, 1369), (1706, 1800)], "red"),
#    "Intersections": ([(103, 118), (267, 285), (429, 446), (602, 632), (792, 811)], "red")
}

intersections = [1, 20, 40, 60, 80, 100, 118]

for label, data in ranges.items():
    for start, stop in data[0]:
        use = dict(x for x in loaded.items() if start <= x[0] <= stop)
        use = use.items()
        use.sort()
        keys = [x[0] for x in use]
        values = [x[1] for x in use]
        plot(keys, values, color = data[1], linewidth=4, label = label)
        label = "_nolegend_"

label = "Intersection"
for intersection in intersections:
    if intersection in loaded:
        plot(intersection, loaded[intersection], "kp", label = label, markersize = 15)
        label = "_nolegend_"

diamondx, diamondy = max(((x[1], x[0]) for x in loaded.items()))
plot(diamondy, diamondx, "k*", label = "Requested Frame", markersize = 20)

xlabel("Frame")
ylabel("Expected Label Change")
#legend(numpoints=1, loc = "best")
ylim(-0.25, 10)
show()

