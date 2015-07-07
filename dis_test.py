def setup(): # executed on each node before jobs are scheduled
    import multiprocessing, multiprocessing.sharedctypes
    # import 'random' into global scope, create global shared variable
    global random, shvar
    import random
    lock = multiprocessing.Lock()
    shvar = multiprocessing.sharedctypes.Value('i', 1, lock=lock)
    return 0

def cleanup(): # executed on each node while cluster is closed
    del globals()['shvar']
    # unload 'random' module (doesn't undo everything import does)
    del globals()['random']

def compute():
    r = random.randint(1, 10)
    global shvar
    shvar.value += r
    return shvar.value

if __name__ == '__main__':
    import dispy
    cluster = dispy.JobCluster(compute, setup=setup, cleanup=cleanup, reentrant=True)
    jobs = []
    for i in range(10):
        job = cluster.submit()
        job.id = i
        jobs.append(job)

    for job in jobs:
        job()
        if job.status != dispy.DispyJob.Finished:
            print('job %s failed: %s' % (job.id, job.exception))
        else:
            print('%s: %s' % (job.id, job.result))