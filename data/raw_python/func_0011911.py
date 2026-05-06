def path(self):
        """
        Path of this node on Studip. Looks like Coures/folder/folder/document. Respects the renaming policies defined in the namemap
        """
        if self.parent is None:
            return self.title
        return join(self.parent.path, self.title)