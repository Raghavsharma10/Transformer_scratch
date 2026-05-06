def decompress_temp_file_object(self, compression):
    """
    <Purpose>
      To decompress a compressed temp file object.  Decompression is performed
      on a temp file object that is compressed, this occurs after downloading
      a compressed file.  For instance if a compressed version of some meta
      file in the repository is downloaded, the temp file containing the
      compressed meta file will be decompressed using this function.
      Note that after calling this method, write() can no longer be called.

                            meta.json.gz
                               |...[download]
                        temporary_file (containing meta.json.gz)
                        /             \
               temporary_file          _orig_file
          containing meta.json          containing meta.json.gz
          (decompressed data)

    <Arguments>
      compression:
        A string indicating the type of compression that was used to compress
        a file.  Only gzip is allowed.

    <Exceptions>
      securesystemslib.exceptions.FormatError: If 'compression' is improperly formatted.

      securesystemslib.exceptions.Error: If an invalid compression is given.

      securesystemslib.exceptions.DecompressionError: If the compression failed for any reason.

    <Side Effects>
      'self._orig_file' is used to store the original data of 'temporary_file'.

    <Return>
      None.
    """

    # Does 'compression' have the correct format?
    # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
    securesystemslib.formats.NAME_SCHEMA.check_match(compression)

    if self._orig_file is not None:
      raise securesystemslib.exceptions.Error('Can only set compression on a'
          ' TempFile once.')

    if compression != 'gzip':
      raise securesystemslib.exceptions.Error('Only gzip compression is'
          ' supported.')

    self.seek(0)
    self._compression = compression
    self._orig_file = self.temporary_file

    try:
      gzip_file_object = gzip.GzipFile(fileobj=self.temporary_file, mode='rb')
      uncompressed_content = gzip_file_object.read()
      self.temporary_file = tempfile.NamedTemporaryFile()
      self.temporary_file.write(uncompressed_content)
      self.flush()

    except Exception as exception:
      raise securesystemslib.exceptions.DecompressionError(exception)