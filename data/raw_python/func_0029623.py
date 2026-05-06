def render(self, data, mimetype):
        """
        Serializes a Python object into a byte array containing a JSON document.
        :param data: A Python object.
        :param mimetype: The mimetype to render the data.
        :return: A byte array containing a JSON document.
        """

        indent = self.__get_indent(mimetype)
        encoding = mimetype.params.get('charset') or 'utf-8'
        return json.dumps(data, indent=indent).encode(encoding)