def _check(self, file):
        """
        Run apropriate check based on `file`'s extension and return it,
        otherwise raise an Error
        """

        if not os.path.exists(file):
            raise Error("file \"{}\" not found".format(file))

        _, extension = os.path.splitext(file)
        try:
            check = self.extension_map[extension[1:]]
        except KeyError:
            magic_type = magic.from_file(file)
            for name, cls in self.magic_map.items():
                if name in magic_type:
                    check = cls
                    break
            else:
                raise Error("unknown file type \"{}\", skipping...".format(file))

        try:
            with open(file) as f:
                code = "\n".join(line.rstrip() for line in f)
        except UnicodeDecodeError:
            raise Error("file does not seem to contain text, skipping...")

        # Ensure we don't warn about adding trailing newline
        try:
            if code[-1] != '\n':
                code += '\n'
        except IndexError:
            pass

        return check(code)