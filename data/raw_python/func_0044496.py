def update(self):
        """Replace baseline representations previously registered for update."""
        for linenum in reversed(sorted(self.updates)):
            self.replace_baseline_repr(linenum, self.updates[linenum])

        if not self.TEST_MODE:
            path = '{}.update{}'.format(*os.path.splitext(self.path))
            with io.open(path, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(self.lines))
            print('UPDATE: {}'.format(self.showpath(path)))