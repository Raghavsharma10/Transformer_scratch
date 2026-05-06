def check(self, results_id):
        """Check for results of a membership request.

        :param str results_id: the ID of a membership request
        :return: successfully created memberships
        :rtype: :class:`list`
        :raises groupy.exceptions.ResultsNotReady: if the results are not ready
        :raises groupy.exceptions.ResultsExpired: if the results have expired
        """
        path = 'results/{}'.format(results_id)
        url = utils.urljoin(self.url, path)
        response = self.session.get(url)
        if response.status_code == 503:
            raise exceptions.ResultsNotReady(response)
        if response.status_code == 404:
            raise exceptions.ResultsExpired(response)
        return response.data['members']