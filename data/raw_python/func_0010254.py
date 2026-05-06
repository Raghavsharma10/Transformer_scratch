def parse_response(cls, response, **kwargs):
        """Parse the server response for this put file command

        This will parse xml of the following form::

            <put_file />

        or with an error::

            <put_file>
                <error ... />
            </put_file>

        :param response: The XML root of the response for a put file command
        :type response: :class:`xml.etree.ElementTree.Element`
        :return: None if everything was ok or an :class:`~ErrorInfo` if the xml contained an error
        """
        if response.tag != cls.command_name:
            raise ResponseParseError(
                "Received response of type {}, PutCommand can only parse responses of type {}".format(response.tag,
                                                                                                      cls.command_name))
        error = response.find('./error')
        if error is not None:
            return _parse_error_tree(error)

        return None