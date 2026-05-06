def set_context_json(self, jsonquery):
        '''
        Get a json parameter and rebuild the context back to a dictionary (probably kwargs)
        '''

        # Make sure we are getting dicts
        if type(jsonquery) != dict:
            raise IOError("set_json_context() method can be called only with dictionaries, you gave me a '{}'".format(type(jsonquery)))

        # Set we will answer json to this request
        self.json = True

        # Transfer keys
        newget = {}
        for key in ['search', 'search_filter_button', 'page', 'pages_to_bring', 'rowsperpage', 'filters', 'year', 'month', 'day', 'hour', 'minute', 'second']:
            if key in jsonquery:
                newget[key] = jsonquery[key]

        # Add transformed ordering
        json_ordering = jsonquery.get('ordering', None)
        if json_ordering:
            # Convert to list
            ordering = []
            for key in json_ordering:
                ordering.append({key: jsonquery['ordering'][key]})

            # Order the result from ordering
            # ordering = sorted(ordering, key=lambda x: abs(x.values()[0]))
            ordering = sorted(ordering, key=lambda x: abs(list(x.values())[0]))
            # Save ordering
            newget['ordering'] = []
            for orderer in ordering:
                key = list(orderer.keys())[0]
                value = orderer[key]
                if value > 0:
                    value = 'asc'
                elif value < 0:
                    value = 'desc'
                else:
                    value = None
                if value:
                    newget['ordering'].append({key: value})

        # Get listid
        newget['listid'] = jsonquery.get("listid", None)

        # Get elementid
        newget['elementid'] = jsonquery.get("elementid", None)

        # Return new get
        return newget