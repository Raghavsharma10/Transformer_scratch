def _val_fs_regex(self, record, hist=None):
        """
        Perform field-specific validation regex

        :param dict record: dictionary of values to validate
        :param dict hist: existing input of history values
        """

        record, hist = self.data_regex_method(fields_list=self.fields,
                                              mongo_db_obj=self.mongo,
                                              hist=hist,
                                              record=record,
                                              lookup_type='fieldSpecificRegex')
        return record, hist