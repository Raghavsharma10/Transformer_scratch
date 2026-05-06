def dataframe(self, predicate=None, filtered_columns=None, columns=None, df_class=None):
        """Return the partition as a Pandas dataframe


        :param predicate: If defined, a callable that is called for each row, and if it returns true, the
        row is included in the output.
        :param filtered_columns: If defined, the value is a dict of column names and
        associated values. Only rows where all of the named columms have the given values will be returned.
        Setting the argument will overwrite any value set for the predicate
        :param columns: A list or tuple of column names to return

        :return: Pandas dataframe

        """

        from operator import itemgetter
        from ambry.pands import AmbryDataFrame

        df_class = df_class or AmbryDataFrame

        if columns:
            ig = itemgetter(*columns)
        else:
            ig = None
            columns = self.table.header

        if filtered_columns:

            def maybe_quote(v):
                from six import string_types
                if isinstance(v, string_types):
                    return '"{}"'.format(v)
                else:
                    return v

            code = ' and '.join("row.{} == {}".format(k, maybe_quote(v))
                                for k, v in filtered_columns.items())

            predicate = eval('lambda row: {}'.format(code))

        if predicate:
            def yielder():
                for row in self.reader:
                    if predicate(row):
                        if ig:
                            yield ig(row)
                        else:
                            yield row.dict

            df = df_class(yielder(), columns=columns, partition=self.measuredim)

            return df

        else:

            def yielder():
                for row in self.reader:
                    yield row.values()

            # Put column names in header order
            columns = [c for c in self.table.header if c in columns]

            return df_class(yielder(), columns=columns, partition=self.measuredim)