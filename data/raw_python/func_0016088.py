def list_apps(self, cmd=None, embed_tasks=False, embed_counts=False,
                  embed_deployments=False, embed_readiness=False,
                  embed_last_task_failure=False, embed_failures=False,
                  embed_task_stats=False, app_id=None, label=None, **kwargs):
        """List all apps.

        :param str cmd: if passed, only show apps with a matching `cmd`
        :param bool embed_tasks: embed tasks in result
        :param bool embed_counts: embed all task counts
        :param bool embed_deployments: embed all deployment identifier
        :param bool embed_readiness: embed all readiness check results
        :param bool embed_last_task_failure: embeds the last task failure
        :param bool embed_failures: shorthand for embed_last_task_failure
        :param bool embed_task_stats: embed task stats in result
        :param str app_id: if passed, only show apps with an 'id' that matches or contains this value
        :param str label: if passed, only show apps with the selected labels
        :param kwargs: arbitrary search filters

        :returns: list of applications
        :rtype: list[:class:`marathon.models.app.MarathonApp`]
        """
        params = {}
        if cmd:
            params['cmd'] = cmd
        if app_id:
            params['id'] = app_id
        if label:
            params['label'] = label

        embed_params = {
            'app.tasks': embed_tasks,
            'app.counts': embed_counts,
            'app.deployments': embed_deployments,
            'app.readiness': embed_readiness,
            'app.lastTaskFailure': embed_last_task_failure,
            'app.failures': embed_failures,
            'app.taskStats': embed_task_stats
        }
        filtered_embed_params = [k for (k, v) in embed_params.items() if v]
        if filtered_embed_params:
            params['embed'] = filtered_embed_params

        response = self._do_request('GET', '/v2/apps', params=params)
        apps = self._parse_response(
            response, MarathonApp, is_list=True, resource_name='apps')
        for k, v in kwargs.items():
            apps = [o for o in apps if getattr(o, k) == v]
        return apps