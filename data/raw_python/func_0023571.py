def _process_response(self, response):
        """Parse response"""

        forward_raw = False
        content_type = response.headers['Content-Type']
        if content_type != 'application/json':
            logger.debug("headers: %s", response.headers)
            # API BUG: text/xml content-type with json payload
            # http://forum.mediafiredev.com/showthread.php?136
            if content_type == 'text/xml':
                # we never request xml, so check it quacks like JSON
                if not response.text.lstrip().startswith('{'):
                    forward_raw = True
            else:
                # _process_response can't deal with non-json,
                # return response as is
                forward_raw = True

        if forward_raw:
            response.raise_for_status()
            return response

        logger.debug("response: %s", response.text)

        # if we are here, then most likely have json
        try:
            response_node = response.json()['response']
        except ValueError:
            # promised JSON but failed
            raise MediaFireApiError("JSON decode failure")

        if response_node.get('new_key', 'no') == 'yes':
            self._regenerate_secret_key()

        # check for errors
        if response_node['result'] != 'Success':
            raise MediaFireApiError(response_node['message'],
                                    response_node['error'])

        return response_node