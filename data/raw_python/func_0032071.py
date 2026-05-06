def itemFromLink(self, link):
        """
        @type link: C{unicode}
        @param link: A webID to translate into an item.

        @rtype: L{Item}
        @return: The item to which the given link referred.
        """
        return self.siteStore.getItemByID(self.webTranslator.linkFrom(link))