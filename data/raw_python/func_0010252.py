def parse_response(cls, response, device_id=None, fssapi=None, **kwargs):
        """Parse the server response for this ls command

        This will parse xml of the following form::

            <ls hash="hash_type">
              <file path="file_path" last_modified=last_modified_time ... />
              ...
              <dir path="dir_path" last_modified=last_modified_time />
              ...
            </ls>

        or with an error::

            <ls>
                <error ... />
            </ls>

        :param response: The XML root of the response for an ls command
        :type response: :class:`xml.etree.ElementTree.Element`
        :param device_id: The device id of the device this ls response came from
        :param fssapi: A reference to a :class:`~FileSystemServiceAPI` for use with the
            :class:`~FileInfo` and :class:`~DirectoryInfo` objects for future commands
        :return: An :class:`~LsInfo` object containing the list of directories and files on
            the device or an :class:`~ErrorInfo` if the xml contained an error
        """
        if response.tag != cls.command_name:
            raise ResponseParseError(
                "Received response of type {}, LsCommand can only parse responses of type {}".format(response.tag,
                                                                                                     cls.command_name))

        if fssapi is None:
            raise FileSystemServiceException("fssapi is required to parse an LsCommand response")
        if device_id is None:
            raise FileSystemServiceException("device_id is required to parse an LsCommand response")

        error = response.find('./error')
        if error is not None:
            return _parse_error_tree(error)

        hash_type = response.get('hash')
        dirs = []
        files = []

        # Get each file listed in this response
        for myfile in response.findall('./file'):
            fi = FileInfo(fssapi,
                          device_id,
                          myfile.get('path'),
                          int(myfile.get('last_modified')),
                          int(myfile.get('size')),
                          myfile.get('hash'),
                          hash_type)
            files.append(fi)
        # Get each directory listed for this device
        for mydir in response.findall('./dir'):
            di = DirectoryInfo(fssapi,
                               device_id,
                               mydir.get('path'),
                               int(mydir.get('last_modified')))
            dirs.append(di)
        return LsInfo(directories=dirs, files=files)