def _lint(self):
        """Run linter in a subprocess."""
        command = self._get_command()
        process = subprocess.run(command, stdout=subprocess.PIPE,  # nosec
                                 stderr=subprocess.PIPE)
        LOG.info('Finished %s', ' '.join(command))
        stdout, stderr = self._get_output_lines(process)
        return self._linter.parse(stdout), self._parse_stderr(stderr)