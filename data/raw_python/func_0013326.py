def _parseAttrs(self, attrsStr):
        """
        Parse the attributes and values
        """
        attributes = dict()
        for attrStr in self.SPLIT_ATTR_COL_RE.split(attrsStr):
            name, vals = self._parseAttrVal(attrStr)
            if name in attributes:
                raise GFF3Exception(
                    "duplicated attribute name: {}".format(name),
                    self.fileName, self.lineNumber)
            attributes[name] = vals
        return attributes