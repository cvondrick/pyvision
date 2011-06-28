import pickle
from pylab import *

loaded = pickle.load(open("probs.pkl"))

set_cmap("gray")
imshow(loaded) 
colorbar()
show()


