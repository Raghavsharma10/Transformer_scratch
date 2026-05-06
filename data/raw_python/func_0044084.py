def run(self, params):
        """Main run method for PyMOC tool.

        Takes a list of command line arguments to process.

        Each operation is performed on a current "running" MOC
        object.
        """

        self.params = list(reversed(params))

        if not self.params:
            self.help()
            return

        while self.params:
            p = self.params.pop()

            if p in self.command:
                # If we got a known command, execute it.
                self.command[p](self)

            elif os.path.exists(p):
                # If we were given the name of an existing file, read it.
                self.read_moc(p)

            else:
                # Otherwise raise an error.
                raise CommandError('file or command {0} not found'.format(p))