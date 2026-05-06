def start(self):
        """
        The main loop, run forever.
        """
        while True:
            self.thread_debug("Interval starting")
            for thr in threading.enumerate():
                self.thread_debug("    " + str(thr))
            self.feed_monitors()
            start = time.time()
            # wait fore queue to empty
            self.workers_queue.join()
            end = time.time()
            diff = self.config['interval']['test'] - (end - start)
            if diff <= 0:
                # alarm
                self.stats.procwin = -diff
                self.thread_debug("Cannot keep up with tests! {} seconds late"
                                  .format(abs(diff)))
            else:
                self.thread_debug("waiting {} seconds...".format(diff))
                time.sleep(diff)