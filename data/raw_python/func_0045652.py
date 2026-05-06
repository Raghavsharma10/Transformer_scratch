def build(self, root="runs"):
        """
        Build a nested directory structure, starting in ``root``

        :param root: Root directory for structure
        """

        for d, control in self.iter(root):
            _mkdirs(d)
            with open(os.path.join(d, self.control_name), 'w') as fp:
                json.dump(control, fp, indent=self.indent)
                # RJSON and some other tools like a trailing newline
                fp.write('\n')