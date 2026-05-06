def query(self, **options):
        '''Return an iterator for the given items.'''
        if not self._query_path:
            raise AttributeError('query is not available for %s' % self._item_name)
        last_item = 0
        offset = 0
        current_item = None
        limit = options.get('limit', 25)
        options['limit'] = limit
        target = self._query_path
        while True:
            options['offset'] = offset
            # go get the data with the given offset
            json_data = self._redmine.get(target, options)
            # Try and read the json
            try:
                data = json.loads(json_data)
            except:
                raise RedmineError(json_data)

            # The data is enclosed in the _query_container item
            # That is, {'issues':{(issue1),(issue2)...}, 'total_count':##}
            data_container = data[self._query_container]
            for item_data in data_container:
                yield(self._objectify(data=item_data))

            # If the container was empty, we requested past the end, just exit
            if not data_container:
                break
            try:
                if int(data['total_count']) > ( offset + len(data_container) ):
                    # moar data!
                    offset += limit
                else:
                    break
            except:
                # If we don't even have a 'total_count', we're done.
                break