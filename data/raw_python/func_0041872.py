def pop_key(self, arg, key, *args, **kwargs):
        """Delete a previously defined key for the `add_argument`
        """
        return self.unfinished_arguments[arg].pop(key, *args, **kwargs)