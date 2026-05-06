def create_deployment(self, ref, force=False, payload='',
                          auto_merge=False, description='', environment=None):
        """Create a deployment.

        :param str ref: (required), The ref to deploy. This can be a branch,
            tag, or sha.
        :param bool force: Optional parameter to bypass any ahead/behind
            checks or commit status checks. Default: False
        :param str payload: Optional JSON payload with extra information about
            the deployment. Default: ""
        :param bool auto_merge: Optional parameter to merge the default branch
            into the requested deployment branch if necessary. Default: False
        :param str description: Optional short description. Default: ""
        :param str environment: Optional name for the target deployment
            environment (e.g., production, staging, qa). Default: "production"
        :returns: :class:`Deployment <github3.repos.deployment.Deployment>`
        """
        json = None
        if ref:
            url = self._build_url('deployments', base_url=self._api)
            data = {'ref': ref, 'force': force, 'payload': payload,
                    'auto_merge': auto_merge, 'description': description,
                    'environment': environment}
            self._remove_none(data)
            headers = Deployment.CUSTOM_HEADERS
            json = self._json(self._post(url, data=data, headers=headers),
                              201)
        return Deployment(json, self) if json else None