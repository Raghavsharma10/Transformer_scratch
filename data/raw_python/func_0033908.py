def run(self):
        """Loops and executes commands in interactive mode."""
        if self._skip_delims:
            delims = readline.get_completer_delims()
            for delim in self._skip_delims:
                delims = delims.replace(delim, '')
            readline.set_completer_delims(delims)

        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._completer.complete)

        if self._history_file:
            # Ensure history file exists
            if not os.path.isfile(self._history_file):
                open(self._history_file, 'w').close()

            readline.read_history_file(self._history_file)

        self._running = True
        try:
            while self._running:
                try:
                    command = input(self._format_prompt())
                    if command:
                        result = self.execute(*shlex.split(command))
                        if result:
                            print(result)
                except CLIException as exc:
                    print(exc)
                except (KeyboardInterrupt, EOFError):
                    self._running = False
                    print()
                except Exception as exc:
                    if self._verbose:
                        traceback.print_exc()
                    else:
                        print(exc)
        finally:
            if self._history_file:
                readline.write_history_file(self._history_file)