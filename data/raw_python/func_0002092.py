def get_xml(self, fp, format=FORMAT_NATIVE):
        """
        Returns the XML metadata for this source, converted to the requested format.
        Converted metadata may not contain all the same information as the native format.

        :param file fp: A path, or an open file-like object which the content should be written to.
        :param str format: desired format for the output. This should be one of the available
            formats from :py:meth:`.get_formats`, or :py:attr:`.FORMAT_NATIVE` for the native format.

        If you pass this function an open file-like object as the fp parameter, the function will
        not close that file for you.
        """
        r = self._client.request('GET', getattr(self, format), stream=True)
        filename = stream.stream_response_to_file(r, path=fp)
        return filename