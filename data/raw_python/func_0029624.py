def __get_indent(self, mimetype):
        """
        Gets the indent parameter from the mimetype.
        :param MimeType mimetype: The mimetype with parameters.
        :return int: The indent if found, otherwise none.
        """
        indent = max(int(mimetype.params.get('indent', '0')), 0)

        if indent == 0:
            return None

        return indent