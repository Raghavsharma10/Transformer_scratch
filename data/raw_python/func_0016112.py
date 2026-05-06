def delete_deployment(self, deployment_id, force=False):
        """Cancel a deployment.

        :param str deployment_id: deployment id
        :param bool force: if true, don't create a rollback deployment to restore the previous configuration

        :returns: a dict containing the deployment id and version (empty dict if force=True)
        :rtype: dict
        """
        if force:
            params = {'force': True}
            self._do_request('DELETE', '/v2/deployments/{deployment}'.format(
                deployment=deployment_id), params=params)
            # Successful DELETE with ?force=true returns empty text (and status
            # code 202). Client code should poll until deployment is removed.
            return {}
        else:
            response = self._do_request(
                'DELETE', '/v2/deployments/{deployment}'.format(deployment=deployment_id))
            return response.json()