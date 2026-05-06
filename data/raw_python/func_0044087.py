def help(self):
        """Display command usage information."""

        if self.params:
            command = self.params.pop().lstrip('-')

            if command in self.command.documentation:
                (aliases, doc) = self.command.documentation[command]
                (synopsis, body) = self._split_docstring(doc)

                print(synopsis)
                if body:
                    print()
                    print(body)

            else:
                raise CommandError('command {0} not known'.format(command))
        else:
            (synopsis, body) = self._split_docstring(__doc__)

            print(synopsis)
            print()
            print(body)
            print()
            print('Commands:')
            for command in sorted(self.command.documentation.keys()):
                print('   ', ', '.join(self.command.documentation[command][0]))
            print()
            print('Use "pymoctool --help COMMAND" for additional '
                  'information about a command.')