def flatten(self):
        """Flatten the scheme into a dictionary where the keys are
        compound 'dot' notation keys, and the values are the corresponding
        options.

        Returns:
            dict: The flattened `Scheme`.
        """
        if self._flat is None:
            flat = {}
            for arg in self.args:
                if isinstance(arg, Option):
                    flat[arg.name] = arg

                elif isinstance(arg, ListOption):
                    flat[arg.name] = arg

                elif isinstance(arg, DictOption):
                    flat[arg.name] = arg
                    if arg.scheme:
                        for k, v in arg.scheme.flatten().items():
                            flat[arg.name + '.' + k] = v

            self._flat = flat
        return self._flat