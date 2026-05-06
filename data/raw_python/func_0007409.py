def rest_put(self, url, params=None, headers=None, auth=None, verify=True, cert=None):
        """
        Perform a PUT request to url with optional authentication
        """
        res = requests.put(url, params=params, headers=headers, auth=auth, verify=verify,
                           cert=cert)
        return res.text, res.status_code