def build(self, parallel=True, debug=False, force=False,
              machine_readable=False):
        """Executes a `packer build`

        :param bool parallel: Run builders in parallel
        :param bool debug: Run in debug mode
        :param bool force: Force artifact output even if exists
        :param bool machine_readable: Make output machine-readable
        """
        self.packer_cmd = self.packer.build

        self._add_opt('-parallel=true' if parallel else None)
        self._add_opt('-debug' if debug else None)
        self._add_opt('-force' if force else None)
        self._add_opt('-machine-readable' if machine_readable else None)
        self._append_base_arguments()
        self._add_opt(self.packerfile)

        return self.packer_cmd()