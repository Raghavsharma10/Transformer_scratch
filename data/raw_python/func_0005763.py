def to_df(self, fields=None, fields_to_explode=None):
        """
        Export items as rows in a pandas dataframe table.

        Parameters
        ----------

        fields: list or dict
            List of field names to export, or dictionary mapping output column names
            to attribute names of the generators.

            Examples:
               fields=['field_name_1', 'field_name_2']
               fields={'COL1': 'field_name_1', 'COL2': 'field_name_2'}

        fields_to_explode: list or None
            Optional list of field names where each entry (which must itself be a sequence)
            is to be "exploded" into separate rows.

        """
        if isinstance(fields, (list, tuple)):
            fields = {name: name for name in fields}

        assert fields_to_explode is None or isinstance(fields_to_explode, (list, tuple))
        if fields_to_explode is None:
            fields_to_explode = []

        if fields is None:
            colnames_to_export = list(self.items[0].as_dict().keys())  # hack! the field names should perhaps be passed in during initialisation?
        else:
            colnames_to_export = list(fields.keys())

        if not set(fields_to_explode).issubset(colnames_to_export):
            raise ValueError(
                "All fields to explode must occur as column names. "
                f"Got field names: {fields_to_explode}. Column names: {list(fields.keys())}"
            )

        if fields is None:
            # New version (much faster!, but needs cleaning up)
            import attr
            df = pd.DataFrame([attr.astuple(x) for x in self.items], columns=colnames_to_export)
            # Old version:
            #return pd.DataFrame([x.to_series() for x in self.items])
        else:
            # New version (much faster!)

            def make_attrgetter(attr_name_new, attr_name, fields_to_explode):
                # TODO: this needs cleaning up!
                if attr_name_new in fields_to_explode and '.' in attr_name:
                    attr_name_first_part, attr_name_rest = attr_name.split('.', maxsplit=1)

                    def func(row):
                        foo_items = attrgetter(attr_name_first_part)(row)
                        return [attrgetter(attr_name_rest)(x) for x in foo_items]

                    return func
                else:
                    return attrgetter(attr_name)

            attr_getters = [make_attrgetter(attr_name_new, attr_name, fields_to_explode) for attr_name_new, attr_name in fields.items()]
            try:
                df = pd.DataFrame([tuple(func(x) for func in attr_getters) for x in self.items], columns=colnames_to_export)
            except AttributeError as exc:
                msg = (
                    "Could not export to dataframe. Did you forget to pass any fields "
                    "which contain sequences within the 'fields_to_explode' argument?. "
                    f"The original error message was: \"{exc}\""
                )
                raise AttributeError(msg)

        if fields_to_explode != []:
            # TODO: add sanity checks to avoid unwanted behaviour (e.g. that all columns
            # to be exploded must have the same number of elements in each entry?)
            df = explode_columns(df, fields_to_explode)

        return df