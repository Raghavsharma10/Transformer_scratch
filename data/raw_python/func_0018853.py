def _prepare_docstrings(self, frame):
        """Assign docstrings to the constants handled by |Constants|
        to make them available in the interactive mode of Python."""
        if config.USEAUTODOC:
            filename = inspect.getsourcefile(frame)
            with open(filename) as file_:
                sources = file_.read().split('"""')
            for code, doc in zip(sources[::2], sources[1::2]):
                code = code.strip()
                key = code.split('\n')[-1].split()[0]
                value = self.get(key)
                if value:
                    value.__doc__ = doc