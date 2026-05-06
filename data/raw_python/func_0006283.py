def get_remote_version(self):
        """ Return the version of the remote Gerrit server. """
        if self.remote_version is None:
            result = self.run_gerrit_command("version")
            version_string = result.stdout.read()
            pattern = re.compile(r'^gerrit version (.*)$')
            self.remote_version = _extract_version(version_string, pattern)
        return self.remote_version