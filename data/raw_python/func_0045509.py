def unschedule(self, task_name):
        '''
        Removes a task from scheduled jobs but it will not kill running tasks
        '''
        for greenlet in self.waiting[task_name]:
            try:
                gevent.kill(greenlet)
            except BaseException:
                pass