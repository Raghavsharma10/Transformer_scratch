def _delete_resource(self, url):
        """
        DELETEs the resource at url
        """
        conn, head = self._construct_request()
        conn.request("DELETE", url, "", head)
        resp = conn.getresponse()
        self._handle_response_errors('DELETE', url, resp)