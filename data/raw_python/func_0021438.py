def _getModelData(self, modelData, parentItem=None):
        """Return the data contained in the model."""
        if parentItem is None:
            parentItem = self.rootItem

        for item in parentItem.getChildren():
            key = item.getItemData(0)
            if item.childCount():
                modelData[key] = odict()
                self._getModelData(modelData[key], item)
            else:
                if isinstance(item.getItemData(2), float):
                    modelData[key] = [item.getItemData(1), item.getItemData(2)]
                else:
                    modelData[key] = item.getItemData(1)