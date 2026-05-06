def close(self):
        '''
        Terminates the session.  It schedules the ``close()`` coroutine
        of the underlying aiohttp session and then enqueues a sentinel
        object to indicate termination.  Then it waits until the worker
        thread to self-terminate by joining.
        '''
        if self._closed:
            return
        self._closed = True
        self._worker_thread.work_queue.put(self.aiohttp_session.close())
        self._worker_thread.work_queue.put(self.worker_thread.sentinel)
        self._worker_thread.join()