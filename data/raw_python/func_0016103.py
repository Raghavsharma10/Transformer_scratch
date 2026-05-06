def kill_given_tasks(self, task_ids, scale=False, force=None):
        """Kill a list of given tasks.

        :param list[str] task_ids: tasks to kill
        :param bool scale: if true, scale down the app by the number of tasks killed
        :param bool force: if true, ignore any current running deployments

        :return: True on success
        :rtype: bool
        """
        params = {'scale': scale}
        if force is not None:
            params['force'] = force
        data = json.dumps({"ids": task_ids})
        response = self._do_request(
            'POST', '/v2/tasks/delete', params=params, data=data)
        return response == 200