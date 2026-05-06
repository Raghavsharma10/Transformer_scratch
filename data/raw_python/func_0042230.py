def obsgroup_generator(self, sql, sql_args):
        """Generator for ObservationGroup

        :param sql:
            A SQL statement which must return rows describing observation groups
        :param sql_args:
            Any variables required to populate the query provided in 'sql'
        :return:
            A generator which produces Event instances from the supplied SQL, closing any opened cursors on completion.
        """

        self.con.execute(sql, sql_args)
        results = self.con.fetchall()
        output = []
        for result in results:
            obs_group = mp.ObservationGroup(group_id=result['publicId'], title=result['title'],
                                            obs_time=result['time'], set_time=result['setAtTime'],
                                            semantic_type=result['semanticType'],
                                            user_id=result['setByUser'])

            # Look up observation group metadata
            sql = """SELECT f.metaKey, stringValue, floatValue
FROM archive_metadata m
INNER JOIN archive_metadataFields f ON m.fieldId=f.uid
WHERE m.groupId=%s
"""
            self.con.execute(sql, (result['uid'],))
            for item in self.con.fetchall():
                value = first_non_null([item['stringValue'], item['floatValue']])
                obs_group.meta.append(mp.Meta(item['metaKey'], value))

            # Fetch observation objects
            sql = """SELECT o.publicId
FROM archive_obs_group_members m
INNER JOIN archive_observations o ON m.observationId=o.uid
WHERE m.groupId=%s
"""
            self.con.execute(sql, (result['uid'],))
            for item in self.con.fetchall():
                obs_group.obs_records.append(self.db.get_observation(item['publicId']))

            output.append(obs_group)

        return output