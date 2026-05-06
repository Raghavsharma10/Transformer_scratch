def unwrap_json(self, type, json_data):
        '''Decodes a json string, and unwraps any 'type' it finds within.'''
        # Parse the data
        try:
            data = json.loads(json_data)
        except ValueError:
            # If parsing failed, then raise the string which likely contains an error message instead of data
            raise RedmineError(json_data)
        # Check to see if there is a data wrapper
        # Some replies will have {'issue':{<data>}} instead of just {<data>}
        try:
            data = data[type]
        except KeyError:
            pass
        return data