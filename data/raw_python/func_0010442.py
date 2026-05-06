def get_field_names(self):
        """
        ACIS web service returns "meta" and "data" for each station;
        Use meta attributes as field names
        """
        field_names = super(StationDataIO, self).get_field_names()
        if set(field_names) == set(['meta', 'data']):
            meta_fields = list(self.data[0]['meta'].keys())
            if set(meta_fields) < set(self.getvalue('meta')):
                meta_fields = self.getvalue('meta')
            field_names = list(meta_fields) + ['data']
        return field_names