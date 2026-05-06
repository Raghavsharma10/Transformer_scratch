def _remove_dead_greenlet(self, task_name):
        '''
        Removes dead greenlet or done task from active list
        '''
        for greenlet in self.active[task_name]:
            try:
                # Allows active greenlet continue to run
                if greenlet.dead:
                    self.active[task_name].remove(greenlet)
            except BaseException:
                pass