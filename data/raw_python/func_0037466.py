def stop(self):
        """ Stop all functions running in the thread handler."""
        for run_event in self.run_events:
            run_event.clear()

        for thread in self.thread_pool:
            thread.join()