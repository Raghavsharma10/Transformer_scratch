def load_data(self, data):
        """
        MultiStnData data results are arrays without explicit dates;
        Infer time series based on start date.
        """

        dates = fill_date_range(self.start_date, self.end_date)
        for row, date in zip(data, dates):
            data = {'date': date}
            if self.add:
                # If self.add is set, results will contain additional
                # attributes (e.g. flags). In that case, create one row per
                # result, with attributes "date", "elem", "value", and one for
                # each item in self.add.
                for elem, vals in zip(self.parameter, row):
                    data['elem'] = elem
                    for add, val in zip(['value'] + self.add, vals):
                        data[add] = val
                    yield data
            else:
                # Otherwise, return one row per date, with "date" and each
                # element's value as attributes.
                for elem, val in zip(self.parameter, row):
                    # namedtuple doesn't like numeric field names
                    if elem.isdigit():
                        elem = "e%s" % elem
                    data[elem] = val
                yield data