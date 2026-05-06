def request(self, action, params=None, action_token_type=None,
                upload_info=None, headers=None):
        """Perform request to MediaFire API

        action -- "category/name" of method to call
        params -- dict of parameters or query string
        action_token_type -- action token to use: None, "upload", "image"
        upload_info -- in case of upload, dict of "fd" and "filename"
        headers -- additional headers to send (used for upload)

        session_token and signature generation/update is handled automatically
        """

        uri = self._build_uri(action)

        if isinstance(params, six.text_type):
            query = params
        else:
            query = self._build_query(uri, params, action_token_type)

        if headers is None:
            headers = {}

        if upload_info is None:
            # Use request body for query
            data = query
            headers['Content-Type'] = FORM_MIMETYPE
        else:
            # Use query string for query since payload is file
            uri += '?' + query

            if "filename" in upload_info:
                data = MultipartEncoder(
                    fields={'file': (
                        upload_info["filename"],
                        upload_info["fd"],
                        UPLOAD_MIMETYPE
                    )}
                )
                headers["Content-Type"] = data.content_type
            else:
                data = upload_info["fd"]
                headers["Content-Type"] = UPLOAD_MIMETYPE

        logger.debug("uri=%s query=%s",
                     uri, query if not upload_info else None)

        try:
            # bytes from now on
            url = (API_BASE + uri).encode('utf-8')
            if isinstance(data, six.text_type):
                # request's data is bytes, dict, or filehandle
                data = data.encode('utf-8')

            response = self.http.post(url, data=data,
                                      headers=headers, stream=True)
        except RequestException as ex:
            logger.exception("HTTP request failed")
            raise MediaFireConnectionError(
                "RequestException: {}".format(ex))

        return self._process_response(response)