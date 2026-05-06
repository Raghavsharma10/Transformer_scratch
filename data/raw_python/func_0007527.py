def bring_to_front(self):
        """TODO: explain depth sorting"""
        if self.parent is not None:
            ch = self.parent.children
            index = ch.index(self)
            ch[-1], ch[index] = ch[index], ch[-1]