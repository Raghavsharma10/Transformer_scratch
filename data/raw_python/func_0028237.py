def dispatch(self, event):
        """
        Dispatch file system event.

        Callback called when there is a file system event. Hooked at 2KGRW.

        This function overrides `FileSystemEventHandler.dispatch`.

        :param event:
            File system event object.

        :return:
            None.
        """
        # Get file path
        file_path = event.src_path

        # If the file path is in extra paths
        if file_path in self._extra_paths:
            # Call `reload`
            self.reload()

        # If the file path ends with `.pyc` or `.pyo`
        if file_path.endswith(('.pyc', '.pyo')):
            # Get `.py` file path
            file_path = file_path[:-1]

        # If the file path ends with `.py`
        if file_path.endswith('.py'):
            # Get the file's directory path
            file_dir = os.path.dirname(file_path)

            # If the file's directory path starts with any of the watch paths
            if file_dir.startswith(tuple(self._watch_paths)):
                # Call `reload`
                self.reload()