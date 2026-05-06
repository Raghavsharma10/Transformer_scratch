def _path_for_file(self, project_name, date):
        """
        Generate the path on disk for a specified project and date.

        :param project_name: the PyPI project name for the data
        :type project: str
        :param date: the date for the data
        :type date: datetime.datetime
        :return: path for where to store this data on disk
        :rtype: str
        """
        return os.path.join(
            self.cache_path,
            '%s_%s.json' % (project_name, date.strftime('%Y%m%d'))
        )