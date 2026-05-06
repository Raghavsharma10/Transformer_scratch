def waitForCompletion (self):
    """Wait for all threads to complete their work

    The worker threads are told to quit when they receive a task
    that is a tuple of (None, None).  This routine puts as many of
    those tuples in the task queue as there are threads.  As soon as
    a thread receives one of these tuples, it dies.
    """
    for x in range(self.numberOfThreads):
      self.taskQueue.put((None, None))
    for t in self.threadList:
      # print "attempting to join %s" % t.getName()
      t.join()