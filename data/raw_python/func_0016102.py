def list_tasks(self, app_id=None, **kwargs):
        """List running tasks, optionally filtered by app_id.

        :param str app_id: if passed, only show tasks for this application
        :param kwargs: arbitrary search filters

        :returns: list of tasks
        :rtype: list[:class:`marathon.models.task.MarathonTask`]
        """
        response = self._do_request(
            'GET', '/v2/apps/%s/tasks' % app_id if app_id else '/v2/tasks')
        tasks = self._parse_response(
            response, MarathonTask, is_list=True, resource_name='tasks')
        [setattr(t, 'app_id', app_id)
         for t in tasks if app_id and t.app_id is None]
        for k, v in kwargs.items():
            tasks = [o for o in tasks if getattr(o, k) == v]

        return tasks