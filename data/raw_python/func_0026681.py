def current_task(self, args):
        """Name of current action for progress-bar output.

        The specific task string is depends on the configuration via `args`.

        Returns
        -------
        ctask : str
            String representation of this task.
        """
        ctask = self.nice_name if self.nice_name is not None else self.name
        if args is not None:
            if args.update:
                ctask = ctask.replace('%pre', 'Updating')
            else:
                ctask = ctask.replace('%pre', 'Loading')
        return ctask