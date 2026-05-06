def rest_get(self, url, params=None, headers=None, auth=None, verify=True, cert=None):
        """
        Perform a GET request to url with optional authentication
        """
        res = requests.get(url, params=params, headers=headers, auth=auth, verify=verify,
                           cert=cert)
        return res.text, res.status_code