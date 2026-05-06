def get_dates_for_project(self, project):
        """
        Return a list of the dates we have in cache for the specified project,
        sorted in ascending date order.

        :param project: project name
        :type project: str
        :return: list of datetime.datetime objects
        :rtype: datetime.datetime
        """
        file_re = re.compile(r'^%s_([0-9]{8})\.json$' % project)
        all_dates = []
        for f in os.listdir(self.cache_path):
            if not os.path.isfile(os.path.join(self.cache_path, f)):
                continue
            m = file_re.match(f)
            if m is None:
                continue
            all_dates.append(datetime.strptime(m.group(1), '%Y%m%d'))
        return sorted(all_dates)