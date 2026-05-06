def set_legend(self, legend):
        """legend needs to be a list, tuple or None"""
        assert(isinstance(legend, list) or isinstance(legend, tuple) or
            legend is None)
        if legend:
            self.legend = [quote(a) for a in legend]
        else:
            self.legend = None