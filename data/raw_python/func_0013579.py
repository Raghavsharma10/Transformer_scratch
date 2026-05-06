def addExpression(self, datafields):
        """
        Adds an Expression to the db.  Datafields is a tuple in the order:
        id, rna_quantification_id, name, expression,
        is_normalized, raw_read_count, score, units, conf_low, conf_hi
        """
        self._expressionValueList.append(datafields)
        if len(self._expressionValueList) >= self._batchSize:
            self.batchAddExpression()