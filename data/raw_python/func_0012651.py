def xmlrpc_get_task(self):
        """
        Return a new task description: ID and necessary parameters, 
        all are given in a dictionary
        """
        try:
            if len(self.reschedule) == 0:
                (task_id, cur_task) = next(self.task_iterator)
            else:
                (task_id, cur_task) = self.reschedule.pop()
            self.scheduled_tasks.update({task_id: cur_task})
            return (task_id, cur_task.to_dict())
        except StopIteration:
            print('StopIteration: No more tasks')
            return False
        except Exception as err:
            print('Some other error')
            print(err)
            return False