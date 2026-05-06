def mp_worker(self):
        """
        Process one task after another by calling the handler (`copy_file` or `copy_link`) method of the super class.
        """
        while not self.task_queue.empty():
            task_kwargs = self.task_queue.get()
            handler_type = task_kwargs.pop('handler_type')

            if handler_type == 'link':
                super(Command, self).link_file(**task_kwargs)
            else:
                super(Command, self).copy_file(**task_kwargs)

            self.task_queue.task_done()