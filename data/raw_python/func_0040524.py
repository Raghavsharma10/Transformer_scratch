def print_results(cls, stdout, stderr):
        """Print linter results and exits with an error if there's any."""
        for line in stderr:
            print(line, file=sys.stderr)
        if stdout:
            if stderr:  # blank line to separate stdout from stderr
                print(file=sys.stderr)
            cls._print_stdout(stdout)
        else:
            print(':) No issues found.')