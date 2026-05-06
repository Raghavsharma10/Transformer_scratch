def _make_request(self, params, translation_url, headers):
        """
            This is the final step, where the request is made, the data is 
            retrieved and returned.
        """
        resp = requests.get(translation_url, params=params, headers=headers)
        resp.encoding = "UTF-8-sig"
        result = resp.json()
        return result