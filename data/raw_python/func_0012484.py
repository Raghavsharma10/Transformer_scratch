def _query_by_system(self, table_name):
        """
        Query for download data broken down by system, for one day.

        :param table_name: table name to query against
        :type table_name: str
        :return: dict of download information by system; keys are project name,
          values are a dict of system names to download count.
        :rtype: dict
        """
        logger.info('Querying for downloads by system in table %s',
                    table_name)
        q = "SELECT file.project, details.system.name, COUNT(*) as dl_count " \
            "%s " \
            "%s " \
            "GROUP BY file.project, details.system.name;" % (
                self._from_for_table(table_name),
                self._where_for_projects
            )
        res = self._run_query(q)
        result = self._dict_for_projects()
        for row in res:
            system = row['details_system_name']
            result[row['file_project']][system] = int(
                row['dl_count'])
        return result