def output_files(self):
        """Returns list of output files from this rule, relative to buildroot.

        In this case it's simple (for now) - the output files are enumerated in
        the rule definition.
        """
        outs = [os.path.join(self.address.repo, self.address.path, x)
                for x in self.params['outs']]
        return outs