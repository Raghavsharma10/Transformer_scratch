def run_plugins(self):
        """
        Creates a thread for each plugin and lets the thread_manager handle it.
        """
        for obj in self.loader.objects:
            # Reserve a slot in the output_dict in order to ensure that the
            # items are in the correct order.
            self.output_dict[obj.output_options['name']] = None
            self.thread_manager.add_thread(obj.main, obj.options['interval'])