def write(self, string: str) -> None:
        """Write the given string as explained in the main documentation
        on class |LogFileInterface|."""
        self.logfile.write('\n'.join(
            f'{self._string}{substring}' if substring else ''
            for substring in string.split('\n')))