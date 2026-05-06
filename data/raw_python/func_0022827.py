def _describe_tree(self, prefix, with_transform):
        """Helper function to actuall construct the tree"""
        extra = ': "%s"' % self.name if self.name is not None else ''
        if with_transform:
            extra += (' [%s]' % self.transform.__class__.__name__)
        output = ''
        if len(prefix) > 0:
            output += prefix[:-3]
            output += '  +--'
        output += '%s%s\n' % (self.__class__.__name__, extra)

        n_children = len(self.children)
        for ii, child in enumerate(self.children):
            sub_prefix = prefix + ('   ' if ii+1 == n_children else '  |')
            output += child._describe_tree(sub_prefix, with_transform)
        return output