def version(self):
        """Software version of the current repository
        """
        branches = self.branches()
        if self.info['branch'] == branches.sandbox:
            try:
                return self.software_version()
            except Exception as exc:
                raise utils.CommandError(
                    'Could not obtain repo version, do you have a makefile '
                    'with version entry?\n%s' % exc
                )
        else:
            branch = self.info['branch'].lower()
            branch = re.sub('[^a-z0-9_-]+', '-', branch)
            return f"{branch}-{self.info['head']['id'][:8]}"