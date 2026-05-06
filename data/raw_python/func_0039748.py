def get_task(self, name, include_helpers=True):
        """
        Get task identified by name or raise TaskNotFound if there
        is no such task
        :param name: name of helper/task to get
        :param include_helpers: if True, also look for helpers
        :return: task or helper identified by name
        """

        if not include_helpers and name in self._helper_names:
            raise TaskNotFound(name)
        try:
            return getattr(self._tasks, name)
        except AttributeError:
            raise TaskNotFound(name)