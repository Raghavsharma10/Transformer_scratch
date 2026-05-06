def addRNAQuantification(self, datafields):
        """
        Adds an RNAQuantification to the db.  Datafields is a tuple in the
        order:
        id, feature_set_ids, description, name,
        read_group_ids, programs, biosample_id
        """
        self._rnaValueList.append(datafields)
        if len(self._rnaValueList) >= self._batchSize:
            self.batchaddRNAQuantification()