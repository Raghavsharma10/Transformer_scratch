def run(self, task):
        '''
        Runs a task and re-schedule it
        '''
        self._remove_dead_greenlet(task.name)
        if isinstance(task.timer, types.GeneratorType):
            # Starts the task immediately
            greenlet_ = gevent.spawn(task.action, *task.args, **task.kwargs)
            self.active[task.name].append(greenlet_)
            try:
                # total_seconds is available in Python 2.7
                greenlet_later = gevent.spawn_later(task.timer.next().total_seconds(), self.run, task)
                self.waiting[task.name].append(greenlet_later)
                return greenlet_, greenlet_later
            except StopIteration:
                pass
            return greenlet_, None
        # Class based timer
        try:
            if task.timer.started is False:
                delay = task.timer.next().total_seconds()
                gevent.sleep(delay)
                greenlet_ = gevent.spawn(task.action, *task.args, **task.kwargs)
                self.active[task.name].append(greenlet_)
            else:
                greenlet_ = gevent.spawn(task.action, *task.args, **task.kwargs)
                self.active[task.name].append(greenlet_)
            greenlet_later = gevent.spawn_later(task.timer.next().total_seconds(), self.run, task)
            self.waiting[task.name].append(greenlet_later)
            return greenlet_, greenlet_later
        except StopIteration:
            pass
        return greenlet_, None