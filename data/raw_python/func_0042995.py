def register_file(self, observation_id, user_id, file_path, file_time, mime_type, semantic_type,
                      file_md5=None, file_meta=None):
        """
        Register a file in the database, also moving the file into the file store. Returns the corresponding FileRecord
        object.

        :param observation_id:
            The publicId of the observation this file belongs to
        :param string user_id:
            The ID of the user who created this file
        :param string file_path:
            The path of the file on disk to register. This file will be moved into the file store and renamed.
        :param string mime_type:
            MIME type of the file
        :param string semantic_type:
            A string defining the semantic type of the file
        :param float file_time:
            UTC datetime of the import of the file into the database
        :param list file_meta:
            A list of :class:`meteorpi_model.Meta` used to provide additional information about this file
        :return:
            The resultant :class:`meteorpi_model.FileRecord` as stored in the database
        """

        if file_meta is None:
            file_meta = []

        # Check that file exists
        if not os.path.exists(file_path):
            raise ValueError('No file exists at {0}'.format(file_path))

        # Get checksum for file, and size
        file_size_bytes = os.stat(file_path).st_size
        file_name = os.path.split(file_path)[1]

        if file_md5 is None:
            file_md5 = mp.get_md5_hash(file_path)

        # Fetch information about parent observation
        self.con.execute("""
SELECT obsTime, l.publicId AS obstory_id, l.name AS obstory_name FROM archive_observations o
INNER JOIN archive_observatories l ON observatory=l.uid
WHERE o.publicId=%s
""", (observation_id,))
        obs = self.con.fetchall()
        if len(obs) == 0:
            raise ValueError("No observation with ID <%s>" % observation_id)
        obs = obs[0]
        repository_fname = mp.get_hash(obs['obsTime'], obs['obstory_id'], file_name)

        # Get ID code for obs_type
        semantic_type_id = self.get_obs_type_id(semantic_type)

        # Insert into database
        self.con.execute("""
INSERT INTO archive_files
(observationId, mimeType, fileName, semanticType, fileTime, fileSize, repositoryFname, fileMD5)
VALUES
((SELECT uid FROM archive_observations WHERE publicId=%s), %s, %s, %s, %s, %s, %s, %s);
""", (observation_id, mime_type, file_name, semantic_type_id, file_time, file_size_bytes, repository_fname, file_md5))

        # Move the original file from its path
        target_file_path = os.path.join(self.file_store_path, repository_fname)
        try:
            shutil.move(file_path, target_file_path)
        except OSError:
            sys.stderr.write("Could not move file into repository\n")

        # Store the file metadata
        for meta in file_meta:
            self.set_file_metadata(user_id, repository_fname, meta, file_time)

        result_file = mp.FileRecord(obstory_id=obs['obstory_id'],
                                    obstory_name=obs['obstory_name'],
                                    observation_id=observation_id,
                                    repository_fname=repository_fname,
                                    file_time=file_time,
                                    file_size=file_size_bytes,
                                    file_name=file_name,
                                    mime_type=mime_type,
                                    semantic_type=semantic_type,
                                    file_md5=file_md5,
                                    meta=file_meta
                                    )

        # Return the resultant file object
        return result_file