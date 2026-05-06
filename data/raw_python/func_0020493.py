def list_builds(self, build_config_id=None, koji_task_id=None,
                    field_selector=None, labels=None):
        """
        List builds matching criteria

        :param build_config_id: str, only list builds created from BuildConfig
        :param koji_task_id: str, only list builds for Koji Task ID
        :param field_selector: str, field selector for query
        :return: HttpResponse
        """
        query = {}
        selector = '{key}={value}'

        label = {}
        if labels is not None:
            label.update(labels)

        if build_config_id is not None:
            label['buildconfig'] = build_config_id

        if koji_task_id is not None:
            label['koji-task-id'] = str(koji_task_id)

        if label:
            query['labelSelector'] = ','.join([selector.format(key=key,
                                                               value=value)
                                               for key, value in label.items()])

        if field_selector is not None:
            query['fieldSelector'] = field_selector
        url = self._build_url("builds/", **query)
        return self._get(url)