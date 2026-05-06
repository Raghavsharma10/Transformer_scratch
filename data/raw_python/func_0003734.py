def dump(self, filename):
        """Dump the registered fields to a file

           Argument:
            | ``filename``  --  the file to write to
        """
        with open(filename, "w") as f:
            for name in sorted(self._fields):
                self._fields[name].dump(f, name)