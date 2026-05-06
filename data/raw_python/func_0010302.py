def get_filedata(self, condition=None, page_size=1000):
        """Return a generator over all results matching the provided condition

        :param condition: An :class:`.Expression` which defines the condition
            which must be matched on the filedata that will be retrieved from
            file data store. If a condition is unspecified, the following condition
            will be used ``fd_path == '~/'``.  This condition will match all file
            data in this accounts "home" directory (a sensible root).
        :type condition: :class:`.Expression` or None
        :param int page_size: The number of results to fetch in a single page.  Regardless
            of the size specified, :meth:`.get_filedata` will continue to fetch pages
            and yield results until all items have been fetched.
        :return: Generator yielding :class:`.FileDataObject` instances matching the
            provided conditions.

        """

        condition = validate_type(condition, type(None), Expression, *six.string_types)
        page_size = validate_type(page_size, *six.integer_types)
        if condition is None:
            condition = (fd_path == "~/")  # home directory

        params = {"embed": "true", "condition": condition.compile()}
        for fd_json in self._conn.iter_json_pages("/ws/FileData", page_size=page_size, **params):
            yield FileDataObject.from_json(self, fd_json)