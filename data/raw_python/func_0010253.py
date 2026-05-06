def parse_response(cls, response, **kwargs):
        """Parse the server response for this get file command

        This will parse xml of the following form::

            <get_file>
                <data>
                   asdfasdfasdfasdfasf
                </data>
            </get_file>

        or with an error::

            <get_file>
                <error ... />
            </get_file>

        :param response: The XML root of the response for a get file command
        :type response: :class:`xml.etree.ElementTree.Element`
        :return: a six.binary_type string of the data of a file or an :class:`~ErrorInfo` if the xml contained an error
        """
        if response.tag != cls.command_name:
            raise ResponseParseError(
                "Received response of type {}, GetCommand can only parse responses of type {}".format(response.tag,
                                                                                                      cls.command_name))

        error = response.find('./error')
        if error is not None:
            return _parse_error_tree(error)

        text = response.find('./data').text
        if text:
            return base64.b64decode(six.b(text))
        else:
            return six.b('')