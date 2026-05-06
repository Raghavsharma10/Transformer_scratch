def rend(self, context, data):
        """
        Automatically retrieve my C{docFactory} based on C{self.fragmentName}
        before invoking L{athena.LiveElement.rend}.
        """
        if self.docFactory is None:
            self.docFactory = self.getDocFactory(self.fragmentName)
        return super(_ThemedMixin, self).rend(context, data)