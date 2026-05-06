def populateFromRow(self, peerRecord):
        """
        This method accepts a model record and sets class variables.
        """
        self.setUrl(peerRecord.url) \
            .setAttributesJson(peerRecord.attributes)
        return self