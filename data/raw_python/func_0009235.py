def __analyze(self):
        """
        Analyze the filter JSON and attempt to parse out the individual filters.
        """
        filter_values = {}

        # ID to filter name mapping
        name_map = {
            10: 'grades',
            11: 'loan_purpose',
            13: 'approved',
            15: 'funding_progress',
            38: 'exclude_existing',
            39: 'term',
            43: 'keyword'
        }

        if self.json is not None:
            filters = self.json

            for f in filters:
                if 'm_id' in f:
                    name = f['m_id']

                    # Get the name to represent this filter
                    if f['m_id'] in name_map:
                        name = name_map[f['m_id']]

                    # Get values
                    if 'm_value' in f:
                        raw_values = f['m_value']
                        value = {}

                        # No value, skip it
                        if raw_values is None:
                            continue

                        # Loop through multiple values
                        if type(raw_values) is list:

                            # A single non string value, is THE value
                            if len(raw_values) == 1 and type(raw_values[0]['value']) not in [str, unicode]:
                                value = raw_values[0]['value']

                            # Create a dict of values: name = True
                            for val in raw_values:
                                if type(val['value']) in [str, unicode]:
                                    value[val['value']] = True

                        # A single value
                        else:
                            value = raw_values

                        # Normalize grades array
                        if name == 'grades':
                            if 'All' not in value:
                                value['All'] = False

                    # Add filter value
                    filter_values[name] = value
                    dict.__setitem__(self, name, value)

        return filter_values