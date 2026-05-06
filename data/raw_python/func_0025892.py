def _objectify(self, json_data=None, data={}):
        '''Return an object derived from the given json data.'''
        if json_data:
            # Parse the data
            try:
                data = json.loads(json_data)
            except ValueError:
                # If parsing failed, then raise the string which likely
                # contains an error message instead of data
                raise RedmineError(json_data)
        # Check to see if there is a data wrapper
        # Some replies will have {'issue':{<data>}} instead of just {<data>}
        try:
            data = data[self._item_type]
        except KeyError:
            pass

        # If there's no ID but a source path
        if ('id' not in data) and ('_source_path' in data):
            # use the path between /projects/ and .json as the ID
            data['id'] = data['_source_path']\
                .partition('/projects/')[2]\
                .partition('.json')[0]

        # Call the base class objectify method
        return super(Redmine_Wiki_Pages_Manager, self)._objectify(data=data)