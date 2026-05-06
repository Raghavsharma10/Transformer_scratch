def path(self):
        """Return the full path to the Group, including any parent Groups."""
        # If root, return '/'
        if self.dataset is self:
            return ''
        else:  # Otherwise recurse
            return self.dataset.path + '/' + self.name