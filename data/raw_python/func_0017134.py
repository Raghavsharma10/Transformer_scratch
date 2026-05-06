def create_status(self, state, target_url='', description=''):
        """Create a new deployment status for this deployment.

        :param str state: (required), The state of the status. Can be one of
            ``pending``, ``success``, ``error``, or ``failure``.
        :param str target_url: The target URL to associate with this status.
            This URL should contain output to keep the user updated while the
            task is running or serve as historical information for what
            happened in the deployment. Default: ''.
        :param str description: A short description of the status. Default: ''.
        :return: partial :class:`DeploymentStatus <DeploymentStatus>`
        """
        json = None

        if state in ('pending', 'success', 'error', 'failure'):
            data = {'state': state, 'target_url': target_url,
                    'description': description}
            response = self._post(self.statuses_url, data=data,
                                  headers=Deployment.CUSTOM_HEADERS)
            json = self._json(response, 201)

        return DeploymentStatus(json, self) if json else None