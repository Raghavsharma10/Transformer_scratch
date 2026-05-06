def _load_compiled(self, file_path):
        """
        Accepts a path to a compiled plugin and returns a module object.

        file_path: A string that represents a complete file path to a compiled
        plugin.
        """
        name = os.path.splitext(os.path.split(file_path)[-1])[0]
        plugin_directory = os.sep.join(os.path.split(file_path)[0:-1])
        compiled_directory = os.path.join(plugin_directory, '__pycache__')
        # Use glob to autocomplete the filename.
        compiled_file = glob.glob(os.path.join(compiled_directory, (name + '.*')))[0]
        plugin = imp.load_compiled(name, compiled_file)
        return plugin