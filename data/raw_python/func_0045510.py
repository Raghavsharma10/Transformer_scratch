def stop_task(self, task_name):
        '''
        Stops a running or dead task
        '''
        for greenlet in self.active[task_name]:
            try:
                # Do not need to check if greenlet is dead, gevent does it already
                gevent.kill(greenlet)
                self.active[task_name] = []
            except BaseException:
                pass