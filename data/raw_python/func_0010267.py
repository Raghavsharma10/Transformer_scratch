def print_subtree(self, fobj=sys.stdout, level=0):
        """Print this group node and the subtree rooted at it"""
        fobj.write("{}{!r}\n".format(" " * (level * 2), self))
        for child in self.get_children():
            child.print_subtree(fobj, level + 1)