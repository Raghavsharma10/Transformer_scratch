def _derive(self, record, hist=None):
        """
        Derivation filters like 'deriveValue' to replace given input values
        from one or more fields. In case 'copyValue' copy value to the target
        field from given an input value from one field. 'deriveRegex' replace
        given an input value from one field, derive target field value using
        regular expressions. If 'deriveIncludes' applies then given an input
        value from one field, derive target field based on at least one of the
        following: includes strings, excludes strings, starts with string,
        ends with string

        :param dict record: dictionary of values to validate
        :param dict hist: existing input of history values
        """

        def check_derive_options(option, derive_set_config):
            """
            Check derive option is exist into options list and return relevant
            flag.
            :param str option: drive options value
            :param list derive_set_config: options list
            :return boolean: True or False based on option exist into options
            list
            """

            return option in derive_set_config

        hist_obj = {}

        if hist is None:
            hist = {}

        for field in record:

            field_val_new = field_val = record[field]

            if field in self.fields:

                for derive_set in self.fields[field]['derive']:

                    check_match = False

                    derive_set_config = derive_set

                    if set.issubset(set(derive_set_config['fieldSet']),
                                    record.keys()):

                        # sorting here to ensure sub document match from
                        # query

                        derive_input = {val: record[val] for val in
                                        derive_set_config['fieldSet']}

                        if derive_set_config['type'] == 'deriveValue':

                            overwrite_flag = check_derive_options(
                                'overwrite',
                                derive_set_config["options"])

                            blank_if_no_match_flag = check_derive_options(
                                'blankIfNoMatch',
                                derive_set_config["options"])

                            field_val_new, hist_obj, check_match = \
                                DeriveDataLookup(
                                    fieldName=field,
                                    db=self.mongo,
                                    deriveInput=derive_input,
                                    overwrite=overwrite_flag,
                                    fieldVal=record[field],
                                    histObj=hist,
                                    blankIfNoMatch=blank_if_no_match_flag)

                        elif derive_set_config['type'] == 'copyValue':

                            overwrite_flag = check_derive_options(
                                'overwrite',
                                derive_set_config["options"])

                            field_val_new, hist_obj, check_match = \
                                DeriveDataCopyValue(
                                    fieldName=field,
                                    deriveInput=derive_input,
                                    overwrite=overwrite_flag,
                                    fieldVal=record[field],
                                    histObj=hist)

                        elif derive_set_config['type'] == 'deriveRegex':

                            overwrite_flag = check_derive_options(
                                'overwrite',
                                derive_set_config["options"])

                            blank_if_no_match_flag = check_derive_options(
                                'blankIfNoMatch',
                                derive_set_config["options"])

                            field_val_new, hist_obj, check_match = \
                                DeriveDataRegex(
                                    fieldName=field,
                                    db=self.mongo,
                                    deriveInput=derive_input,
                                    overwrite=overwrite_flag,
                                    fieldVal=record[field],
                                    histObj=hist,
                                    blankIfNoMatch=blank_if_no_match_flag)

                        elif derive_set_config['type'] == 'deriveIncludes':

                            overwrite_flag = check_derive_options(
                                'overwrite',
                                derive_set_config["options"])

                            blank_if_no_match_flag = check_derive_options(
                                'blankIfNoMatch',
                                derive_set_config["options"])

                            field_val_new, hist_obj, check_match = \
                                IncludesLookup(
                                    fieldVal=record[field],
                                    lookupType='deriveIncludes',
                                    deriveFieldName= \
                                        derive_set_config['fieldSet'][0],
                                    deriveInput=derive_input,
                                    db=self.mongo,
                                    fieldName=field,
                                    histObj=hist,
                                    overwrite=overwrite_flag,
                                    blankIfNoMatch=blank_if_no_match_flag)

                    if check_match or field_val_new != field_val:
                        record[field] = field_val_new
                        break

        return record, hist_obj