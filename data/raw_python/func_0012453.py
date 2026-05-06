def get(self, project, date):
        """
        Get the cache data for a specified project for the specified date.
        Returns None if the data cannot be found in the cache.

        :param project: PyPi project name to get data for
        :type project: str
        :param date: date to get data for
        :type date: datetime.datetime
        :return: dict of per-date data for project
        :rtype: :py:obj:`dict` or ``None``
        """
        fpath = self._path_for_file(project, date)
        logger.debug('Cache GET project=%s date=%s - path=%s',
                     project, date.strftime('%Y-%m-%d'), fpath)
        try:
            with open(fpath, 'r') as fh:
                data = json.loads(fh.read())
        except:
            logger.debug('Error getting from cache for project=%s date=%s',
                         project, date.strftime('%Y-%m-%d'))
            return None
        data['cache_metadata']['date'] = datetime.strptime(
            data['cache_metadata']['date'],
            '%Y%m%d'
        )
        data['cache_metadata']['updated'] = datetime.fromtimestamp(
            data['cache_metadata']['updated']
        )
        return data