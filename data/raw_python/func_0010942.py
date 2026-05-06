def attributes(self):
        """ Returns all attributes in the schema """
        attrs = self._schema["attributes"]
        return [item_attribute(attr) for attr in sorted(attrs.values(),
                key=operator.itemgetter("defindex"))]