def hook(self, id_num):
        """Get a single hook.

        :param int id_num: (required), id of the hook
        :returns: :class:`Hook <github3.repos.hook.Hook>` if successful,
            otherwise None
        """
        json = None
        if int(id_num) > 0:
            url = self._build_url('hooks', str(id_num), base_url=self._api)
            json = self._json(self._get(url), 200)
        return Hook(json, self) if json else None