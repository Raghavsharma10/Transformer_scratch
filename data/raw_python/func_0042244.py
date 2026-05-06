def _norm_lookup(self, record, hist=None):
        """
        Perform generic validation lookup

        :param dict record: dictionary of values to validate
        :param dict hist: existing input of history values
        """

        record, hist = self.data_lookup_method(fields_list=self.fields,
                                               mongo_db_obj=self.mongo,
                                               hist=hist,
                                               record=record,
                                               lookup_type='normLookup')
        return record, hist