def run(command, input=None, exit=0, shell=False):
        """
        Run `command` passing it stdin from `input`, throwing a DependencyError if comand is not found.
        Throws Error if exit code of command is not `exit` (unless `exit` is None).
        """
        if isinstance(input, str):
            input = input.encode()

        # Only pipe stdin if we have input to pipe.
        stdin = {} if input is None else {"stdin": subprocess.PIPE}
        try:
            child = subprocess.Popen(command, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, **stdin)
        except FileNotFoundError as e:
            # Extract name of command.
            name = command.split(' ', 1)[0] if isinstance(command, str) else command[0]
            raise DependencyError(name)

        stdout, _ = child.communicate(input=input)
        if exit is not None and child.returncode != exit:
            raise Error("failed to stylecheck code")
        return stdout.decode()