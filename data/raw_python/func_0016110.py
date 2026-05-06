def list_deployments(self):
        """List all running deployments.

        :returns: list of deployments
        :rtype: list[:class:`marathon.models.deployment.MarathonDeployment`]
        """
        response = self._do_request('GET', '/v2/deployments')
        return self._parse_response(response, MarathonDeployment, is_list=True)