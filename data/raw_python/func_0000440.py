def clone(self, name=None):
        """
        :param name: new env name
        :rtype: Environment
        """
        resp = self._router.post_env_clone(env_id=self.environmentId, json=dict(name=name) if name else {}).json()
        return Environment(self.organization, id=resp['id']).init_router(self._router)