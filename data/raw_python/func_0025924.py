def _objectify(self, json_data=None, data={}):
        '''Return an object derived from the given json data.'''
        if json_data:
            # Parse the data
            try:
                data = json.loads(json_data)
            except ValueError:
                # If parsing failed, then raise the string which likely contains an error message instead of data
                raise RedmineError(json_data)
        # Check to see if there is a data wrapper
        # Some replies will have {'issue':{<data>}} instead of just {<data>}
        try:
            data = data[self._item_type]
        except KeyError:
            pass

        # Either returns a new item or updates the item in the cache and returns that
        return self._redmine.check_cache(self._item_type, data, self._object)