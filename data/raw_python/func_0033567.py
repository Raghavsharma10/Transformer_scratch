def body(self):
        """Gets the JSON body of the request"""

        if self._decoded_body == None:
            # Try to decode the JSON body. But raise an error if the
            # content-type is unexpected, or the JSON is invalid.

            raw_content_type = self.request.headers.get("content-type") or ""
            content_type = raw_content_type.split(";")[0].strip().lower()

            if content_type == "application/json":
                try:
                    self._decoded_body = escape.json_decode(self.request.body)
                except:
                    raise oz.json_api.ApiError("Bad JSON body")
            else:
                raise oz.json_api.ApiError("JSON body expected")

        return self._decoded_body