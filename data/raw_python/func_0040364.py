def feed_monitors(self):
        """
        Pull from the cached monitors data and feed the workers queue.  Run
        every interval (refresh:test).
        """
        self.thread_debug("Filling worker queue...", module='feed_monitors')
        for mon in self.monitors:
            self.thread_debug("    Adding " + mon['title'])
            self.workers_queue.put(mon)