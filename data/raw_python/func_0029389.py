def run(self, name):
        """Runs the function associated with the given entry `name`."""
        for entry in self.entries:
            if entry.name == name:
                run_func(entry)
                break