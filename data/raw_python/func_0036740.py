def _do_api_call(self, method, data):
        """
        Convenience method to carry out a standard API call against the
        Petfinder API.

        :param basestring method: The API method name to call.
        :param dict data: Key/value parameters to send to the API method.
            This varies based on the method.
        :raises: A number of :py:exc:`petfinder.exceptions.PetfinderAPIError``
            sub-classes, depending on what went wrong.
        :rtype: lxml.etree._Element
        :returns: The parsed document.
        """

        # Developer API keys, auth tokens, and other standard, required args.
        data.update({
            "key": self.api_key,
            # No API methods currently use this, but we're ready for it,
            # should that change.
            "token": self.api_auth_token,
        })

        # Ends up being a full URL+path.
        url = "%s%s" % (self.endpoint, method)
        # Bombs away!
        response = requests.get(url, params=data)

        # Parse and return an ElementTree instance containing the document.
        root = etree.fromstring(response.content)

        # If this is anything but '100', it's an error.
        status_code = root.find("header/status/code").text
        # If this comes back as non-None, we know we've got problems.
        exc_class = _get_exception_class_from_status_code(status_code)
        if exc_class:
            # Sheet, sheet, errar! Raise the appropriate error, and pass
            # the accompanying error message as the exception message.
            error_message = root.find("header/status/message").text
            #noinspection PyCallingNonCallable
            raise exc_class(error_message)

        return root