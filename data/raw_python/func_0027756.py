def _massageData(self, row):

        """
        Convert a row into a tuple of Item instances, by slicing it
        according to the number of columns for each instance, and then
        proceeding as for ItemQuery._massageData.

        @param row: an n-tuple, where n is the total number of columns
        specified by all the item types in this query.

        @return: a tuple of instances of the types specified by this query.
        """
        offset = 0
        resultBits = []

        for i, tableClass in enumerate(self.tableClass):
            numAttrs = self.schemaLengths[i]

            result = self.store._loadedItem(self.tableClass[i],
                                            row[offset],
                                            row[offset+1:offset+numAttrs])
            assert result.store is not None, "result %r has funky store" % (result,)
            resultBits.append(result)

            offset += numAttrs

        return tuple(resultBits)