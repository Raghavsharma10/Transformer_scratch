def list_tasks(self):
        """
        Get the tasks of current object

        :return: the tasks
        :rtype: list
        """
        from highton.models.task import Task

        return fields.ListField(
            name=self.ENDPOINT,
            init_class=Task
        ).decode(
            self.element_from_string(
                self._get_request(
                    endpoint=self.ENDPOINT + '/' + str(self.id) + '/' + Task.ENDPOINT,
                ).text
            )
        )