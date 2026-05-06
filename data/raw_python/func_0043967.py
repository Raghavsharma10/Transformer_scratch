def create(self, fullname, shortname, category_id, **kwargs):
        """
        Create a new course

        :param string fullname: The course's fullname
        :param string shortname: The course's shortname
        :param int category_id: The course's category

        :keyword string idnumber: (optional) Course ID number. \
            Yes, it's a string, blame Moodle.
        :keyword int summaryformat: (optional) Defaults to 1 (HTML). \
            Summary format options: (1 = HTML, 0 = Moodle, 2 = Plain, \
            or 4 = Markdown)
        :keyword string format: (optional) Defaults to "topics"
            Topic options: (weeks, topics, social, site)
        :keyword bool showgrades: (optional) Defaults to True. \
            Determines if grades are shown
        :keyword int newsitems: (optional) Defaults to 5. \
            Number of recent items appearing on the course page
        :keyword bool startdate: (optional) Timestamp when the course start
        :keyword int maxbytes: (optional) Defaults to 83886080. \
            Largest size of file that can be uploaded into the course
        :keyword bool showreports: Default to True. Are activity report shown?
        :keyword bool visible: (optional) Determines if course is \
            visible to students
        :keyword int groupmode: (optional) Defaults to 2.
            options: (0 = no group, 1 = separate, 2 = visible)
        :keyword bool groupmodeforce: (optional) Defaults to False. \
            Force group mode
        :keyword int defaultgroupingid: (optional) Defaults to 0. \
            Default grouping id
        :keyword bool enablecompletion: (optional) Enable control via \
            completion in activity settings.
        :keyword bool completionstartonenrol: (optional) \
            Begin tracking a student's progress in course completion after
        :keyword bool completionnotify: (optional) Default? Dunno. \
            Presumably notifies course completion
        :keyword string lang: (optional) Force course language.
        :keyword string forcetheme: (optional) Name of the force theme

        Example Usage::

        >>> import muddle
        >>> muddle.course().create('a new course', 'new-course', 20)
        """

        allowed_options = ['idnumber', 'summaryformat',
                           'format', 'showgrades',
                           'newsitems', 'startdate',
                           'maxbytes', 'showreports',
                           'visible', 'groupmode',
                           'groupmodeforce', 'jdefaultgroupingid',
                           'enablecompletion', 'completionstartonenrol',
                           'completionnotify', 'lang',
                           'forcetheme']

        if valid_options(kwargs, allowed_options):
            option_params = {}
            for index, key in enumerate(kwargs):
                val = kwargs.get(key)

                if isinstance(val, bool):
                    val = int(val)

                option_params.update({'courses[0][' + key + ']': val})

            params = {'wsfunction': 'core_course_create_courses',
                      'courses[0][fullname]': fullname,
                      'courses[0][shortname]': shortname,
                      'courses[0][categoryid]': category_id}

            params.update(option_params)
            params.update(self.request_params)

            return requests.post(self.api_url, params=params, verify=False)