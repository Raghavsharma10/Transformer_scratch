def getNodesCheckState(self, parentItem=None):
        """Return the check state (disabled, tristate, enable) of all items
        belonging to a parent.
        """
        if parentItem is None:
            parentItem = self.rootItem

        checkStates = odict()
        children = parentItem.getChildren()

        for child in children:
            checkStates[child.itemData[0]] = child.getCheckState()

        return checkStates