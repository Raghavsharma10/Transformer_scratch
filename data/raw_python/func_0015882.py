def push(self, create=True, token=False):
        """Implmenets the `packer push` function

        UNTESTED! Must be used alongside an Atlas account
        """
        self.packer_cmd = self.packer.push

        self._add_opt('-create=true' if create else None)
        self._add_opt('-tokn={0}'.format(token) if token else None)
        self._add_opt(self.packerfile)

        return self.packer_cmd()