def _parseAttrVal(self, attrStr):
        """
        Returns tuple of tuple of (attr, value), multiple are returned to
        handle multi-value attributes.
        """
        m = self.SPLIT_ATTR_RE.match(attrStr)
        if m is None:
            raise GFF3Exception(
                "can't parse attribute/value: '" + attrStr +
                "'", self.fileName, self.lineNumber)
        name = urllib.unquote(m.group(1))
        val = m.group(2)
        # Split by comma to separate then unquote.
        # Commas in values must be url encoded.
        return name, [urllib.unquote(v) for v in val.split(',')]