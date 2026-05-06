def render_docstring(self):
        """make a nice docstring for ipython"""
        res = '{{{self.method}}} {self.uri} {self.title}\n'.format(self=self)
        if self.params:
            for group, params in self.params.items():
                res += '\n' + group + ' params:\n'
                for param in params.values():
                    res += param.render_docstring()
        return res