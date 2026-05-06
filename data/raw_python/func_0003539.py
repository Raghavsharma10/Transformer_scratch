def queryTypesDescriptions(self, types):
        """
        Given a list of types, construct a dictionary such that
        each key is a type, and each value is the corresponding sObject
        for that type.
        """
        types = list(types)
        if types:
            types_descs = self.describeSObjects(types)
        else:
            types_descs = []
        return dict(map(lambda t, d: (t, d), types, types_descs))