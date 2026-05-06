def _list_clouds(self):
        """
        Request a list of all added clouds.

        Populates self._clouds dict with mist.client.model.Cloud instances
        """
        req = self.request(self.uri + '/clouds')
        clouds = req.get().json()
        if clouds:
            for cloud in clouds:
                self._clouds[cloud['id']] = Cloud(cloud, self)
        else:
            self._clouds = {}