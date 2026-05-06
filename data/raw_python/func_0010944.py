def attributes(self):
        """ Returns a list of attributes """

        overridden_attrs = self._attributes
        sortmap = {"neutral": 1, "positive": 2,
                   "negative": 3}

        sortedattrs = list(overridden_attrs.values())
        sortedattrs.sort(key=operator.itemgetter("defindex"))
        sortedattrs.sort(key=lambda t: sortmap.get(t.get("effect_type",
                                                         "neutral"), 99))
        return [item_attribute(theattr) for theattr in sortedattrs]