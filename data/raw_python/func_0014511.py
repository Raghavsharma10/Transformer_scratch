def _list_available_rest_versions(self):
        """Return a list of the REST API versions supported by the array"""
        url = "https://{0}/api/api_version".format(self._target)

        data = self._request("GET", url, reestablish_session=False)
        return data["version"]