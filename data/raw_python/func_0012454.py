def set(self, project, date, data, data_ts):
        """
        Set the cache data for a specified project for the specified date.

        :param project: project name to set data for
        :type project: str
        :param date: date to set data for
        :type date: datetime.datetime
        :param data: data to cache
        :type data: dict
        :param data_ts: maximum timestamp in the BigQuery data table
        :type data_ts: int
        """
        data['cache_metadata'] = {
            'project': project,
            'date': date.strftime('%Y%m%d'),
            'updated': time.time(),
            'version': VERSION,
            'data_ts': data_ts
        }
        fpath = self._path_for_file(project, date)
        logger.debug('Cache SET project=%s date=%s - path=%s',
                     project, date.strftime('%Y-%m-%d'), fpath)
        with open(fpath, 'w') as fh:
            fh.write(json.dumps(data))