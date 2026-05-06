def rest_del(self, url, params=None, auth=None, verify=True, cert=None):
        """
        Perform a DELETE request to url with optional authentication
        """
        res = requests.delete(url, params=params, auth=auth, verify=verify, cert=cert)
        return res.text, res.status_code