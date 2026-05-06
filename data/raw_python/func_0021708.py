def depth(self):
        """Increase the level of indentation by one."""
        if self.indentation is None:
            yield
        else:
            previous = self.previous_indent
            self.previous_indent = self.indent
            self.indent += self.indentation
            yield
            self.indent = self.previous_indent
            self.previous_indent = previous