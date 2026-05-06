def observation_generator(self, sql, sql_args):
        """Generator for Observation

        :param sql:
            A SQL statement which must return rows describing observations
        :param sql_args:
            Any variables required to populate the query provided in 'sql'
        :return:
            A generator which produces Event instances from the supplied SQL, closing any opened cursors on completion.
        """

        self.con.execute(sql, sql_args)
        results = self.con.fetchall()
        output = []
        for result in results:
            observation = mp.Observation(obstory_id=result['obstory_id'], obstory_name=result['obstory_name'],
                                         obs_time=result['obsTime'], obs_id=result['publicId'],
                                         obs_type=result['obsType'])

            # Look up observation metadata
            sql = """SELECT f.metaKey, stringValue, floatValue
FROM archive_metadata m
INNER JOIN archive_metadataFields f ON m.fieldId=f.uid
WHERE m.observationId=%s
"""
            self.con.execute(sql, (result['uid'],))
            for item in self.con.fetchall():
                value = first_non_null([item['stringValue'], item['floatValue']])
                observation.meta.append(mp.Meta(item['metaKey'], value))

            # Fetch file objects
            sql = "SELECT f.repositoryFname FROM archive_files f WHERE f.observationId=%s"
            self.con.execute(sql, (result['uid'],))
            for item in self.con.fetchall():
                observation.file_records.append(self.db.get_file(item['repositoryFname']))

            # Count votes for observation
            self.con.execute("SELECT COUNT(*) FROM archive_obs_likes WHERE observationId="
                             "(SELECT uid FROM archive_observations WHERE publicId=%s);", (result['publicId'],))
            observation.likes = self.con.fetchone()['COUNT(*)']

            output.append(observation)

        return output