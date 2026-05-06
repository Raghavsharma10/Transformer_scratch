def kill_task(self, app_id, task_id, scale=False, wipe=False):
        """Kill a task.

        :param str app_id: application ID
        :param str task_id: the task to kill
        :param bool scale: if true, scale down the app by one if the task exists

        :returns: the killed task
        :rtype: :class:`marathon.models.task.MarathonTask`
        """
        params = {'scale': scale, 'wipe': wipe}
        response = self._do_request('DELETE', '/v2/apps/{app_id}/tasks/{task_id}'
                                    .format(app_id=app_id, task_id=task_id), params)
        # Marathon is inconsistent about what type of object it returns on the multi
        # task deletion endpoint, depending on the version of Marathon. See:
        # https://github.com/mesosphere/marathon/blob/06a6f763a75fb6d652b4f1660685ae234bd15387/src/main/scala/mesosphere/marathon/api/v2/AppTasksResource.scala#L88-L95
        if "task" in response.json():
            return self._parse_response(response, MarathonTask, is_list=False, resource_name='task')
        else:
            return response.json()