def _responsive_join(self, thread, waiting_func=None):
        """similar to the responsive sleep, a join function blocks a thread
        until some other thread dies.  If that takes a long time, we'd like to
        have some indicaition as to what the waiting thread is doing.  This
        method will wait for another thread while calling the waiting_func
        once every second.

        parameters:
            thread - an instance of the TaskThread class representing the
                     thread to wait for
            waiting_func - a function to call every second while waiting for
                           the thread to die"""
        while True:
            try:
                thread.join(1.0)
                if not thread.isAlive():
                    break
                if waiting_func:
                    waiting_func()
            except KeyboardInterrupt:
                self.logger.debug('quit detected by _responsive_join')
                self.quit = True