def get_export_configurations(self):
        """
        Retrieve all ExportConfigurations held in this db

        :return: a list of all :class:`meteorpi_model.ExportConfiguration` on this server
        """
        sql = (
            'SELECT uid, exportConfigId, exportType, searchString, targetURL, '
            'targetUser, targetPassword, exportName, description, active '
            'FROM archive_exportConfig ORDER BY uid DESC')
        return list(self.generators.export_configuration_generator(sql=sql, sql_args=[]))