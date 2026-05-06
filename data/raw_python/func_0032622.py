def childFactory(self, ctx, name):
        """
        Return a shell page wrapped around the Item model described by the
        webID, or return None if no such item can be found.
        """
        try:
            o = self.webapp.fromWebID(name)
        except _WebIDFormatException:
            return None
        if o is None:
            return None
        return self.webViewer.wrapModel(o)