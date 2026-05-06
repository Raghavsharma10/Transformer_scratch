def map_value(self, field, value):
        """
        Clean up some values returned from the web service.
        (overrides wq.io.mappers.BaseMapper)
        """

        if field == 'sids':
            # Site identifiers are returned as "[id] [auth_id]";
            # Map to auth name for easier usability
            ids = {}
            for idinfo in value:
                id, auth = idinfo.split(' ')
                auth = AUTHORITY_BY_ID[auth]
                ids[auth['name']] = id
            return ids

        if field == 'valid_daterange':
            # Date ranges for each element are returned in an array
            # (sorted by the order the elements were were requested);
            # Convert to dictionary with element id as key
            elems, complex = self.getlist('parameter')
            ranges = {}
            for elem, val in zip(elems, value):
                if val:
                    start, end = val
                    ranges[elem] = (parse_date(start), parse_date(end))
                else:
                    ranges[elem] = None, None
            return ranges
        return value