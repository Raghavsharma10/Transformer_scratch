def to_dict(self, files=None):
        """
        Converts the CodeBaseDoc into a dictionary containing the to_dict()
        representations of each contained file.  The optional `files` list
        lets you restrict the dict to include only specific files.

        >>> CodeBaseDoc(['examples']).to_dict(['class.js']).get('module.js')
        >>> CodeBaseDoc(['examples']).to_dict(['class.js'])['class.js'][0]['name']
        'MyClass'

        """
        keys = files or list(self.keys())
        return dict((key, self[key].to_dict()) for key in keys)