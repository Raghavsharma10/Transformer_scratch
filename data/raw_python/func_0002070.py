def validate(self, export):
        """
        Validates an Export.


        :param Export export:
        :rtype: ExportValidationResponse
        """
        target_url = self.client.get_url(self._URL_KEY, 'POST', 'validate')
        response_object = ExportValidationResponse()
        r = self.client.request('POST', target_url, json=export._serialize())
        return response_object._deserialize(r.json())