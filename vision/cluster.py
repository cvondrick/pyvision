from multiprocessing import Process, cpu_count
from multiprocessing.managers import BaseManager
import Queue

port = 4242
authkey = 'pyvision'

class QueueManager(BaseManager): 
    pass

problems = None
solutions = None
def setup(server):
    global problems, solutions
    QueueManager.register('get_problems')
    QueueManager.register('get_solutions')
    m = QueueManager(address=(server, port), authkey = authkey)
    m.connect()
    problems = m.get_problems()
    solutions = m.get_solutions()

def ask(token, *args, **kwargs):
    problems.put((token, args, kwargs))

data = {}
def answer(wanted):
    if wanted in data:
        return data.pop(wanted)
    while True:
        token, resp = solutions.get()
        if token == wanted:
            return resp
        data[token] = resp

def getproblem():
    return problems.get()

def solveproblem(token, resp):
    solutions.put((token, resp))

def imap(it):
    for i, n in enumerate(it):
        ask(i, n)
    for i in range(len(it)):
        yield answer(i)

def map(it):
    return list(imap(it))

class Worker(Process):
    def __init__(self, callable):
        self.callable = callable
        super(Worker, self).__init__()
    def run(self):
        self.callable()

def handler(callable):
    workers = []
    for i in range(cpu_count()):
        w = Worker(callable)
        w.start()
        workers.append(w)
    for worker in workers:
        worker.join()

def start_server():
    problems = Queue.Queue()
    solutions = Queue.Queue()
    QueueManager.register('get_problems', callable = lambda: problems)
    QueueManager.register('get_solutions', callable = lambda: solutions)
    m = QueueManager(address=('', port), authkey = authkey)
    s = m.get_server()
    s.serve_forever()


if __name__ == "__main__":
    print "Started server."
    start_server()
