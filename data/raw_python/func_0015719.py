def write_into(self, block, level=0):
        """Append this block to another one, passing all dependencies"""

        for line, l in self._lines:
            block.write_line(line, level + l)

        for name, obj in _compat.iteritems(self._deps):
            block.add_dependency(name, obj)