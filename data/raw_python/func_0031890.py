def post_result_data(self, client, check, output, status):
        """
        Posts check result data.
        """
        data = {
            'source': client,
            'name': check,
            'output': output,
            'status': status,
        }
        self._request('POST', '/results', data=json.dumps(data))
        return True