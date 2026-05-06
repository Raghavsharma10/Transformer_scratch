def add_file(self, fp, upload_path=None, content_type=None):
        """
        Add a single file or archive to upload.

        To add metadata records with a file, add a .xml file with the same upload path basename
        eg. ``points-with-metadata.geojson`` & ``points-with-metadata.xml``
        Datasource XML must be in one of these three formats:

          - ISO 19115/19139
          - FGDC CSDGM
          - Dublin Core (OAI-PMH)

        :param fp: File to upload into this source, can be a path or a file-like object.
        :type fp: str or file
        :param str upload_path: relative path to store the file as within the source (eg. ``folder/0001.tif``). \
                                By default it will use ``fp``, either the filename from a path or the ``.name`` \
                                attribute of a file-like object.
        :param str content_type: Content-Type of the file. By default it will attempt to auto-detect from the \
                                 file/upload_path.
        """
        if isinstance(fp, six.string_types):
            # path
            if not os.path.isfile(fp):
                raise ClientValidationError("Invalid file: %s", fp)
            if not upload_path:
                upload_path = os.path.split(fp)[1]
        else:
            # file-like object
            if not upload_path:
                upload_path = os.path.split(fp.name)[1]

        content_type = content_type or mimetypes.guess_type(upload_path, strict=False)[0]
        if upload_path in self._files:
            raise ClientValidationError("Duplicate upload path: %s" % upload_path)

        self._files[upload_path] = (fp, content_type)
        logger.debug("UploadSource.add_file: %s -> %s (%s)", repr(fp), upload_path, content_type)