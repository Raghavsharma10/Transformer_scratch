def get_aggregate_check(self, check, age=None):
        """
        Returns the list of aggregates for a given check
        """
        data = {}
        if age:
            data['max_age'] = age

        result = self._request('GET', '/aggregates/{}'.format(check),
                               data=json.dumps(data))
        return result.json()