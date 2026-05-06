def mark_entities_to_export(self, export_config):
        """
        Apply the specified :class:`meteorpi_model.ExportConfiguration` to the database, running its contained query and
        creating rows in t_observationExport or t_fileExport for matching entities.

        :param ExportConfiguration export_config:
            An instance of :class:`meteorpi_model.ExportConfiguration` to apply.
        :returns:
            The integer number of rows added to the export tables
        """
        # Retrieve the internal ID of the export configuration, failing if it hasn't been stored
        self.con.execute('SELECT uid FROM archive_exportConfig WHERE exportConfigID = %s;',
                         (export_config.config_id,))
        export_config_id = self.con.fetchall()
        if len(export_config_id) < 1:
            raise ValueError("Attempt to run export on ExportConfiguration not in database")
        export_config_id = export_config_id[0]['uid']

        # If the export is inactive then do nothing
        if not export_config.enabled:
            return 0

        # Track the number of rows created, return it later
        rows_created = 0

        # Handle ObservationSearch
        if isinstance(export_config.search, mp.ObservationSearch):
            # Create a deep copy of the search and set the properties required when creating exports
            search = mp.ObservationSearch.from_dict(export_config.search.as_dict())
            search.exclude_export_to = export_config.config_id
            b = search_observations_sql_builder(search)

            self.con.execute(b.get_select_sql(columns='o.uid, o.obsTime'), b.sql_args)
            for result in self.con.fetchall():
                self.con.execute('INSERT INTO archive_observationExport '
                                 '(observationId, obsTime, exportConfig, exportState) '
                                 'VALUES (%s,%s,%s,%s)', (result['uid'], result['obsTime'], export_config_id, 1))
                rows_created += 1

        # Handle FileSearch
        elif isinstance(export_config.search, mp.FileRecordSearch):
            # Create a deep copy of the search and set the properties required when creating exports
            search = mp.FileRecordSearch.from_dict(export_config.search.as_dict())
            search.exclude_export_to = export_config.config_id
            b = search_files_sql_builder(search)

            self.con.execute(b.get_select_sql(columns='f.uid, f.fileTime'), b.sql_args)
            for result in self.con.fetchall():
                self.con.execute('INSERT INTO archive_fileExport '
                                 '(fileId, fileTime, exportConfig, exportState) '
                                 'VALUES (%s,%s,%s,%s)', (result['uid'], result['fileTime'], export_config_id, 1))
                rows_created += 1

        # Handle ObservatoryMetadataSearch
        elif isinstance(export_config.search, mp.ObservatoryMetadataSearch):
            # Create a deep copy of the search and set the properties required when creating exports
            search = mp.ObservatoryMetadataSearch.from_dict(export_config.search.as_dict())
            search.exclude_export_to = export_config.config_id
            b = search_metadata_sql_builder(search)

            self.con.execute(b.get_select_sql(columns='m.uid, m.setAtTime'), b.sql_args)
            for result in self.con.fetchall():
                self.con.execute('INSERT INTO archive_metadataExport '
                                 '(metadataId, setAtTime, exportConfig, exportState) '
                                 'VALUES (%s,%s,%s,%s)', (result['uid'], result['setAtTime'], export_config_id, 1))
                rows_created += 1

        # Complain if it's anything other than these two (nothing should be at the moment but we might introduce
        # more search types in the future
        else:
            raise ValueError("Unknown search type %s" % str(type(export_config.search)))
        return rows_created