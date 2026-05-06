def create_hook(self, name, config, events=['push'], active=True):
        """Create a hook on this repository.

        :param str name: (required), name of the hook
        :param dict config: (required), key-value pairs which act as settings
            for this hook
        :param list events: (optional), events the hook is triggered for
        :param bool active: (optional), whether the hook is actually
            triggered
        :returns: :class:`Hook <github3.repos.hook.Hook>` if successful,
            otherwise None
        """
        json = None
        if name and config and isinstance(config, dict):
            url = self._build_url('hooks', base_url=self._api)
            data = {'name': name, 'config': config, 'events': events,
                    'active': active}
            json = self._json(self._post(url, data=data), 201)
        return Hook(json, self) if json else None