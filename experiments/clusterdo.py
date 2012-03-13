from vision import cluster

def isprime(n):
    for i in xrange(2, n):
        if n % i == 0:
            return False
    return True

cluster.setup('quickstep')

@cluster.handler
def launch():
    while True:
        token, problem, _ = cluster.getproblem()
        result = isprime(problem[0])
        cluster.solveproblem(token, result)
