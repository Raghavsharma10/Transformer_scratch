def file_handler(self, handler_type, path, prefixed_path, source_storage):
        """
        Create a dict with all kwargs of the `copy_file` or `link_file` method of the super class and add it to
        the queue for later processing.
        """
        if self.faster:
            if prefixed_path not in self.found_files:
                self.found_files[prefixed_path] = (source_storage, path)

            self.task_queue.put({
                'handler_type': handler_type,
                'path': path,
                'prefixed_path': prefixed_path,
                'source_storage': source_storage
            })
            self.counter += 1
        else:
            if handler_type == 'link':
                super(Command, self).link_file(path, prefixed_path, source_storage)
            else:
                super(Command, self).copy_file(path, prefixed_path, source_storage)