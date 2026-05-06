def usable_item(self, data):
        """
        ACIS web service returns "meta" and "data" for each station; use meta
        attributes as item values, and add an IO for iterating over "data"
        """

        # Use metadata as item
        item = data['meta']

        # Add nested IO for data
        elems, elems_is_complex = self.getlist('parameter')
        if elems_is_complex:
            elems = [elem['name'] for elem in elems]

        add, add_is_complex = self.getlist('add')
        item['data'] = DataIO(
            data=data['data'],
            parameter=elems,
            add=add,
            start_date=self.getvalue('start_date'),
            end_date=self.getvalue('end_date'),
        )

        # TupleMapper will convert item to namedtuple
        return super(StationDataIO, self).usable_item(item)