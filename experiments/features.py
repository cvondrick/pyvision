from vision import features
from PIL import Image
from scipy.io import savemat as savematlab

im = Image.open("/scratch/vatic/syn-bounce-level/0/0/0.jpg")
f = features.hog(im)
savematlab(open("features.mat", "w"), {"py": f})
