def to_df(self, fields=None):
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
        """
        if isinstance(fields, (list, tuple)):
            fields = {name: name for name in fields}

        if fields is None:
            # New version (much faster!, but needs cleaning up)
            import attr
            colnames = list(self.items[0].as_dict().keys())  # hack! the field names should perhaps be passed in during initialisation?
            return pd.DataFrame([attr.astuple(x) for x in self.items], columns=colnames)
            # Old version:
            #return pd.DataFrame([x.to_series() for x in self.items])
        else:
            # New version (much faster!)
            colnames = list(fields.keys())
            attr_getters = [attrgetter(attr_name) for attr_name in fields.values()]
            return pd.DataFrame([tuple(func(x) for func in attr_getters) for x in self.items], columns=colnames)