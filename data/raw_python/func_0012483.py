def _query_by_installer(self, table_name):
        """
        Query for download data broken down by installer, for one day.

        :param table_name: table name to query against
        :type table_name: str
        :return: dict of download information by installer; keys are project
          name, values are a dict of installer names to dicts of installer
          version to download count.
        :rtype: dict
        """
        logger.info('Querying for downloads by installer in table %s',
                    table_name)
        q = "SELECT file.project, details.installer.name, " \
            "details.installer.version, COUNT(*) as dl_count " \
            "%s " \
            "%s " \
            "GROUP BY file.project, details.installer.name, " \
            "details.installer.version;" % (
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
            iname = row['details_installer_name']
            iver = row['details_installer_version']
            if iname not in proj:
                proj[iname] = {}
            if iver not in proj[iname]:
                proj[iname][iver] = 0
            proj[iname][iver] += int(row['dl_count'])
        return result