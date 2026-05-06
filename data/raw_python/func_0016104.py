def kill_tasks(self, app_id, scale=False, wipe=False,
                   host=None, batch_size=0, batch_delay=0):
        """Kill all tasks belonging to app.

        :param str app_id: application ID
        :param bool scale: if true, scale down the app by the number of tasks killed
        :param str host: if provided, only terminate tasks on this Mesos slave
        :param int batch_size: if non-zero, terminate tasks in groups of this size
        :param int batch_delay: time (in seconds) to wait in between batched kills. If zero, automatically determine

        :returns: list of killed tasks
        :rtype: list[:class:`marathon.models.task.MarathonTask`]
        """
        def batch(iterable, size):
            sourceiter = iter(iterable)
            while True:
                batchiter = itertools.islice(sourceiter, size)
                yield itertools.chain([next(batchiter)], batchiter)

        if batch_size == 0:
            # Terminate all at once
            params = {'scale': scale, 'wipe': wipe}
            if host:
                params['host'] = host
            response = self._do_request(
                'DELETE', '/v2/apps/{app_id}/tasks'.format(app_id=app_id), params)
            # Marathon is inconsistent about what type of object it returns on the multi
            # task deletion endpoint, depending on the version of Marathon. See:
            # https://github.com/mesosphere/marathon/blob/06a6f763a75fb6d652b4f1660685ae234bd15387/src/main/scala/mesosphere/marathon/api/v2/AppTasksResource.scala#L88-L95
            if "tasks" in response.json():
                return self._parse_response(response, MarathonTask, is_list=True, resource_name='tasks')
            else:
                return response.json()
        else:
            # Terminate in batches
            tasks = self.list_tasks(
                app_id, host=host) if host else self.list_tasks(app_id)
            for tbatch in batch(tasks, batch_size):
                killed_tasks = [self.kill_task(app_id, t.id, scale=scale, wipe=wipe)
                                for t in tbatch]

                # Pause until the tasks have been killed to avoid race
                # conditions
                killed_task_ids = set(t.id for t in killed_tasks)
                running_task_ids = killed_task_ids
                while killed_task_ids.intersection(running_task_ids):
                    time.sleep(1)
                    running_task_ids = set(
                        t.id for t in self.get_app(app_id).tasks)

                if batch_delay == 0:
                    # Pause until the replacement tasks are healthy
                    desired_instances = self.get_app(app_id).instances
                    running_instances = 0
                    while running_instances < desired_instances:
                        time.sleep(1)
                        running_instances = sum(
                            t.started_at is None for t in self.get_app(app_id).tasks)
                else:
                    time.sleep(batch_delay)

            return tasks