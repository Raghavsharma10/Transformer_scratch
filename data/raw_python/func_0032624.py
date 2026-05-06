def linkToWithActiveTab(self, childItem, parentItem):
        """
        Return a URL which will point to the web facet of C{childItem},
        with the selected nav tab being the one that represents C{parentItem}
        """
        return self.linkTo(parentItem.storeID) + '/' + self.toWebID(childItem)