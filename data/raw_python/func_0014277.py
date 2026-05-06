def construct_include(self, node):
        """Include file referenced at node."""

        filename = os.path.join(self._root, self.construct_scalar(node))
        filename = os.path.abspath(filename)
        extension = os.path.splitext(filename)[1].lstrip('.')

        with open(filename, 'r') as f:
            if extension in ('yaml', 'yml'):
                return yaml.load(f, Loader=self)
            else:
                return ''.join(f.readlines())