def get_activities_for_project(self, module=None, **kwargs):
        """Get the related activities of a project.

        :param str module: Stages of a given module
        :return: JSON
        """

        _module_id = kwargs.get('module', module)
        _activities_url = ACTIVITIES_URL.format(module_id=_module_id)
        return self._request_api(url=_activities_url).json()