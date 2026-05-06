def fix(self, to_file=None):
        """Implements the `packer fix` function

        :param string to_file: File to output fixed template to
        """
        self.packer_cmd = self.packer.fix

        self._add_opt(self.packerfile)

        result = self.packer_cmd()
        if to_file:
            with open(to_file, 'w') as f:
                f.write(result.stdout.decode())
        result.fixed = json.loads(result.stdout.decode())
        return result