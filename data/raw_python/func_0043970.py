def duplicate(self, fullname, shortname, categoryid,
                  visible=True, **kwargs):
        """
        Duplicates an existing course with options.
        Note: Can be very slow running.

        :param string fullname: The new course's full name
        :param string shortname: The new course's short name
        :param string categoryid: Category new course should be created under

        :keyword bool visible: Defaults to True. The new course's visiblity
        :keyword bool activities: (optional) Defaults to True. \
            Include course activites
        :keyword bool blocks: (optional) Defaults to True. \
            Include course blocks
        :keyword bool filters: (optional) Defaults to True. \
            Include course filters
        :keyword bool users: (optional) Defaults to False. Include users
        :keyword bool role_assignments: (optional) Defaults to False. \
            Include role assignments
        :keyword bool comments: (optional) Defaults to False. \
            Include user comments
        :keyword bool usercompletion: (optional) Defaults to False. \
            Include user course completion information
        :keyword bool logs: (optional) Defaults to False. Include course logs
        :keyword bool grade_histories: (optional) Defaults to False. \
            Include histories

        :returns: response object

        Example Usage::

        >>> import muddle
        >>> muddle.course(10).duplicate('new-fullname', 'new-shortname', 20)
        """

        # TODO
        # Ideally categoryid should be optional here and
        # should default to catid of course being duplicated.

        allowed_options = ['activities', 'blocks',
                           'filters', 'users',
                           'role_assignments', 'comments',
                           'usercompletion', 'logs',
                           'grade_histories']

        if valid_options(kwargs, allowed_options):
            option_params = {}
            for index, key in enumerate(kwargs):
                option_params.update(
                    {'options[' + str(index) + '][name]': key,
                     'options[' + str(index) + '][value]':
                        int(kwargs.get(key))})

            params = {'wsfunction': 'core_course_duplicate_course',
                      'courseid': self.course_id,
                      'fullname': fullname,
                      'shortname': shortname,
                      'categoryid': categoryid,
                      'visible': int(visible)}
            params.update(option_params)
            params.update(self.request_params)

            return requests.post(self.api_url, params=params, verify=False)