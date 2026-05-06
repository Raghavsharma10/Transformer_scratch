def get_group_for_activity(self, module=None, project=None, **kwargs):
        """Get groups for activity.

        :param str module: Base module
        :param str module: Project which contains the group requested
        :return: JSON
        """

        _module_id = kwargs.get('module', module)
        _project_id = kwargs.get('project', project)
        _url = GROUPS_URL.format(module_id=_module_id, project_id=_project_id)
        return self._request_api(url=_url).json()