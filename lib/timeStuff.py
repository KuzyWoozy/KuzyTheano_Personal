import time

def timeIt(processName):
    def timeIt_withArgs(func):
        def newFunc(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            print("{0:s} took {1:.2f} secs".format(processName, time.time()-start))
        return newFunc
    return timeIt_withArgs



