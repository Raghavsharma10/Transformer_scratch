def get_app(self, app_id, embed_tasks=False, embed_counts=False,
                embed_deployments=False, embed_readiness=False,
                embed_last_task_failure=False, embed_failures=False,
                embed_task_stats=False):
        """Get a single app.

        :param str app_id: application ID
        :param bool embed_tasks: embed tasks in result
        :param bool embed_counts: embed all task counts
        :param bool embed_deployments: embed all deployment identifier
        :param bool embed_readiness: embed all readiness check results
        :param bool embed_last_task_failure: embeds the last task failure
        :param bool embed_failures: shorthand for embed_last_task_failure
        :param bool embed_task_stats: embed task stats in result

        :returns: application
        :rtype: :class:`marathon.models.app.MarathonApp`
        """
        params = {}
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

        response = self._do_request(
            'GET', '/v2/apps/{app_id}'.format(app_id=app_id), params=params)
        return self._parse_response(response, MarathonApp, resource_name='app')