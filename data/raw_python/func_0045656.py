def complete(self):
        """
        Complete current task
        :return:
        :rtype: requests.models.Response
        """
        return self._post_request(
            data='',
            endpoint=self.ENDPOINT + '/' + str(self.id) + '/complete'
        )