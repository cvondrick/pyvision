from vision import cluster

cluster.setup('quickstep')

start = 1000000000
length = 10000
jobs = range(start, start + length)

print "asking"
for job in jobs:
    cluster.ask(job, job)

print "answering"
for job in jobs:
    print job, cluster.answer(job)
