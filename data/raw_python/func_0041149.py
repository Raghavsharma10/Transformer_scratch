def url(self):
        """The concatenation of the `base_url` and `end_url` that make up the
        resultant url.

        :return: the `base_url` and the `end_url`.
        :rtype: str
        """
        end_url = ("/accounts/{account_id}/alerts/{alert_id}/mentions/"
                  "{mention_id}/children?")

        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}

        keys = {"access_token", "account_id", "alert_id"}
        parameters = without_keys(self.params, keys)

        for key, value in list(parameters.items()):
            if value != '':
                end_url += '&' + key + '={' + key + '}'

        end_url = end_url.format(**self.params)
        return self._base_url + end_url