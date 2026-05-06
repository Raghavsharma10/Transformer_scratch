def get_next_entity_to_export(self):
        """
        Examines the archive_observationExport and archive_metadataExport tables, and builds
        either a :class:`meteorpi_db.ObservationExportTask` or a :class:`meteorpi_db.MetadataExportTask` as appropriate.
        These task objects can be used to retrieve the underlying entity and export configuration, and to update the
        completion state or push the timestamp into the future, deferring evaluation of the task until later.

        :returns:
            Either None, if no exports are available, or an object, depending on whether an observation or metadata
            item is next in the queue to export.
        """

        # If the queue of items waiting to export is old, delete it and fetch a new list from the database
        if self.export_queue_valid_until < time.time():
            self.export_queue_metadata = []
            self.export_queue_observations = []
            self.export_queue_files = []

        # If we don't have a queue of items waiting to export, query database for items
        if (not self.export_queue_metadata) and (not self.export_queue_observations) and (not self.export_queue_files):
            self.export_queue_valid_until = time.time() + 60

            # Try to retrieve the earliest record in archive_metadataExport
            self.con.execute('SELECT c.exportConfigId, o.publicId, x.exportState, '
                             'c.targetURL, c.targetUser, c.targetPassword '
                             'FROM archive_metadataExport x '
                             'INNER JOIN archive_exportConfig c ON x.exportConfig=c.uid '
                             'INNER JOIN archive_metadata o ON x.metadataId=o.uid '
                             'WHERE c.active = 1 AND x.exportState > 0 '
                             'ORDER BY x.setAtTime ASC, o.uid ASC LIMIT 50')
            self.export_queue_metadata = list(self.con.fetchall())

            if not self.export_queue_metadata:

                # Try to retrieve the earliest record in archive_observationExport
                self.con.execute('SELECT c.exportConfigId, o.publicId, x.exportState, '
                                 'c.targetURL, c.targetUser, c.targetPassword '
                                 'FROM archive_observationExport x '
                                 'INNER JOIN archive_exportConfig c ON x.exportConfig=c.uid '
                                 'INNER JOIN archive_observations o ON x.observationId=o.uid '
                                 'WHERE c.active = 1  AND x.exportState > 0 '
                                 'ORDER BY x.obsTime ASC, o.uid ASC LIMIT 50')
                self.export_queue_observations = list(self.con.fetchall())

                if not self.export_queue_observations:
                    # Try to retrieve the earliest record in archive_fileExport
                    self.con.execute('SELECT c.exportConfigId, o.repositoryFname, x.exportState, '
                                     'c.targetURL, c.targetUser, c.targetPassword '
                                     'FROM archive_fileExport x '
                                     'INNER JOIN archive_exportConfig c ON x.exportConfig=c.uid '
                                     'INNER JOIN archive_files o ON x.fileId=o.uid '
                                     'WHERE c.active = 1 AND x.exportState > 0 '
                                     'ORDER BY x.fileTime ASC, o.uid ASC LIMIT 50')
                    self.export_queue_files = list(self.con.fetchall())

        if self.export_queue_metadata:
            row = self.export_queue_metadata.pop(0)
            config_id = row['exportConfigId']
            entity_id = row['publicId']
            status = row['exportState']
            target_url = row['targetURL']
            target_user = row['targetUser']
            target_password = row['targetPassword']
            return MetadataExportTask(db=self, config_id=config_id, metadata_id=entity_id,
                                      status=status, target_url=target_url, target_user=target_user,
                                      target_password=target_password)

        if self.export_queue_observations:
            row = self.export_queue_observations.pop(0)
            config_id = row['exportConfigId']
            entity_id = row['publicId']
            status = row['exportState']
            target_url = row['targetURL']
            target_user = row['targetUser']
            target_password = row['targetPassword']
            return ObservationExportTask(db=self, config_id=config_id, observation_id=entity_id,
                                         status=status, target_url=target_url, target_user=target_user,
                                         target_password=target_password)

        if self.export_queue_files:
            row = self.export_queue_files.pop(0)
            config_id = row['exportConfigId']
            entity_id = row['repositoryFname']
            status = row['exportState']
            target_url = row['targetURL']
            target_user = row['targetUser']
            target_password = row['targetPassword']
            return FileExportTask(db=self, config_id=config_id, file_id=entity_id,
                                  status=status, target_url=target_url, target_user=target_user,
                                  target_password=target_password)

        return None