def write_file(self, path, name, data, content_type=None, archive=False,
                   raw=False):
        """Write a file to the file data store at the given path

        :param str path: The path (directory) into which the file should be written.
        :param str name: The name of the file to be written.
        :param data: The binary data that should be written into the file.
        :type data: str (Python2) or bytes (Python3)
        :param content_type: The content type for the data being written to the file.  May
             be left unspecified.
        :type content_type: str or None
        :param bool archive: If true, history will be retained for various revisions of this
            file.  If this is not required, leave as false.
        :param bool raw: If true, skip the FileData XML headers (necessary for binary files)

        """
        path = validate_type(path, *six.string_types)
        name = validate_type(name, *six.string_types)
        data = validate_type(data, six.binary_type)
        content_type = validate_type(content_type, type(None), *six.string_types)
        archive_str = "true" if validate_type(archive, bool) else "false"

        if not path.startswith("/"):
            path = "/" + path
        if not path.endswith("/"):
            path += "/"
        name = name.lstrip("/")

        sio = six.moves.StringIO()
        if not raw:
            if six.PY3:
                base64_encoded_data = base64.encodebytes(data).decode('utf-8')
            else:
                base64_encoded_data = base64.encodestring(data)

            sio.write("<FileData>")
            if content_type is not None:
                sio.write("<fdContentType>{}</fdContentType>".format(content_type))
            sio.write("<fdType>file</fdType>")
            sio.write("<fdData>{}</fdData>".format(base64_encoded_data))
            sio.write("<fdArchive>{}</fdArchive>".format(archive_str))
            sio.write("</FileData>")
        else:
            sio.write(data)

        params = {
            "type": "file",
            "archive": archive_str
        }
        self._conn.put(
            "/ws/FileData{path}{name}".format(path=path, name=name),
            sio.getvalue(),
            params=params)