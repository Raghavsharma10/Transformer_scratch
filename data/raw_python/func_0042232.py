def export_configuration_generator(self, sql, sql_args):
        """
        Generator for :class:`meteorpi_model.ExportConfiguration`

        :param sql:
            A SQL statement which must return rows describing export configurations
        :param sql_args:
            Any variables required to populate the query provided in 'sql'
        :return:
            A generator which produces :class:`meteorpi_model.ExportConfiguration` instances from the supplied SQL,
            closing any opened cursors on completion.
        """

        self.con.execute(sql, sql_args)
        results = self.con.fetchall()
        output = []
        for result in results:
            if result['exportType'] == "observation":
                search = mp.ObservationSearch.from_dict(json.loads(result['searchString']))
            elif result['exportType'] == "file":
                search = mp.FileRecordSearch.from_dict(json.loads(result['searchString']))
            else:
                search = mp.ObservatoryMetadataSearch.from_dict(json.loads(result['searchString']))
            conf = mp.ExportConfiguration(target_url=result['targetURL'], user_id=result['targetUser'],
                                          password=result['targetPassword'], search=search,
                                          name=result['exportName'], description=result['description'],
                                          enabled=result['active'], config_id=result['exportConfigId'])
            output.append(conf)

        return output