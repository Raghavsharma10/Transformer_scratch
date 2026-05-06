def index(cls):
        """Fetches all records.

        Returns:
            `dict`. The JSON formatted response.

        Raises:
            `requests.exceptions.HTTPError`: The status code is not ok.
        """
        res = requests.get(cls.URL, headers=HEADERS, verify=False)
        res.raise_for_status()
        return res.json()