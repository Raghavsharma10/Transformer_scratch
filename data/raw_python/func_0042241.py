def data_lookup_method(fields_list, mongo_db_obj, hist, record,
                           lookup_type):
        """
        Method to lookup the replacement value given a single input value from
        the same field.

        :param dict fields_list: Fields configurations
        :param MongoClient mongo_db_obj: MongoDB collection object
        :param dict hist: existing input of history values object
        :param dict record: values to validate
        :param str lookup_type: Type of lookup
        """

        if hist is None:
            hist = {}

        for field in record:

            if record[field] != '' and record[field] is not None:

                if field in fields_list:

                    if lookup_type in fields_list[field]['lookup']:

                        field_val_new, hist = DataLookup(
                            fieldVal=record[field],
                            db=mongo_db_obj,
                            lookupType=lookup_type,
                            fieldName=field,
                            histObj=hist)

                        record[field] = field_val_new

        return record, hist