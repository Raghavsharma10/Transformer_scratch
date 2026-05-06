def unroot(self):
        """
        Unroots current node. This function is expected to be used on
        the absolute tree root node, but it can be also be applied to
        any other internal node. It will convert a split into a
        multifurcation.
        """
        if len(self.children)==2:
            if not self.children[0].is_leaf():
                self.children[0].delete()
            elif not self.children[1].is_leaf():
                self.children[1].delete()
            else:
                raise TreeError("Cannot unroot a tree with only two leaves")