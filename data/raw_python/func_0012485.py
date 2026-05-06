def _query_by_distro(self, table_name):
        """
        Query for download data broken down by OS distribution, for one day.

        :param table_name: table name to query against
        :type table_name: str
        :return: dict of download information by distro; keys are project name,
          values are a dict of distro names to dicts of distro version to
          download count.
        :rtype: dict
        """
        logger.info('Querying for downloads by distro in table %s', table_name)
        q = "SELECT file.project, details.distro.name, " \
            "details.distro.version, COUNT(*) as dl_count " \
            "%s " \
            "%s " \
            "GROUP BY file.project, details.distro.name, " \
            "details.distro.version;" % (
                self._from_for_table(table_name),
                self._where_for_projects
            )
        res = self._run_query(q)
        result = self._dict_for_projects()
        # iterate through results
        for row in res:
            # pointer to the per-project result dict
            proj = result[row['file_project']]
            # grab the name and version; change None to 'unknown'
            dname = row['details_distro_name']
            dver = row['details_distro_version']
            if dname not in proj:
                proj[dname] = {}
            if dver not in proj[dname]:
                proj[dname][dver] = 0
            proj[dname][dver] += int(row['dl_count'])
        return result