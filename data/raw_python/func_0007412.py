def rest_get_stream(self, url, auth=None, verify=True, cert=None):
        """
        Perform a chunked GET request to url with optional authentication
        This is specifically to download files.
        """
        res = requests.get(url, auth=auth, stream=True, verify=verify, cert=cert)
        return res.raw, res.status_code