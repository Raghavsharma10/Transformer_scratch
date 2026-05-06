def obstory_metadata_generator(self, sql, sql_args):
        """
        Generator for :class:`meteorpi_model.CameraStatus`

        :param sql:
            A SQL statement which must return rows describing obstory metadata
        :param sql_args:
            Any arguments required to populate the query provided in 'sql'
        :return:
            A generator which produces :class:`meteorpi_model.CameraStatus` instances from the supplied SQL, closing
            any opened cursors on completion
        """

        self.con.execute(sql, sql_args)
        results = self.con.fetchall()
        output = []
        for result in results:
            value = ""
            if ('floatValue' in result) and (result['floatValue'] is not None):
                value = result['floatValue']
            if ('stringValue' in result) and (result['stringValue'] is not None):
                value = result['stringValue']
            obs_meta = mp.ObservatoryMetadata(metadata_id=result['metadata_id'], obstory_id=result['obstory_id'],
                                              obstory_name=result['obstory_name'],
                                              obstory_lat=result['obstory_lat'], obstory_lng=result['obstory_lng'],
                                              key=result['metadata_key'], value=value,
                                              metadata_time=result['time'], time_created=result['time_created'],
                                              user_created=result['user_created'])
            output.append(obs_meta)

        return output