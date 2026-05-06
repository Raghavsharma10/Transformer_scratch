def featureName(self):
        """
        ID attribute from GFF3 or None if record doesn't have it.
        Called "Name" rather than "Id" within GA4GH, as there is
        no guarantee of either uniqueness or existence.
        """
        featId = self.attributes.get("ID")
        if featId is not None:
            featId = featId[0]
        return featId