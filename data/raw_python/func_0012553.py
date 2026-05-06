def main_loop(self, stop_at_empty=False):
        """Blocking function that can be run directly, if so would probably
        want to specify 'stop_at_empty' to true, or have a separate process
        adding items to the queue. """
        try:
            while True:
                self.hook_pre_command()
                self._check_command_queue()
                if self.run_until and self.run_until < datetime.datetime.now():
                    self.log.info("Time limit reached")
                    break
                if self._end.value:
                    break
                if self._pause.value:
                    time.sleep(.5)
                    continue
                self.hook_post_command()
                self._update_tasks()
                task_id = self._free_task()
                if task_id:
                    try:
                        task = self.task_queue.get(timeout=.1)
                    except queue.Empty:
                        if stop_at_empty:
                            break
                        self._return_task(task_id)
                    else:
                        self.hook_pre_task()
                        self.log.debug("Starting task on {0}".format(task_id))
                        try:
                            self._start_task(task_id, task)
                        except Exception as err:
                            self.log.exception("Could not start task {0} -"
                                               " {1}".format(task_id, err))
                        else:
                            self.hook_post_task()
        finally:
            self.log.info("Ending main loop")