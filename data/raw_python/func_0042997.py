def get_observation(self, observation_id):
        """
        Retrieve an existing :class:`meteorpi_model.Observation` by its ID

        :param string observation_id:
            UUID of the observation
        :return:
            A :class:`meteorpi_model.Observation` instance, or None if not found
        """
        search = mp.ObservationSearch(observation_id=observation_id)
        b = search_observations_sql_builder(search)
        sql = b.get_select_sql(columns='l.publicId AS obstory_id, l.name AS obstory_name, '
                                       'o.obsTime, s.name AS obsType, o.publicId, o.uid',
                               skip=0, limit=1, order='o.obsTime DESC')
        obs = list(self.generators.observation_generator(sql=sql, sql_args=b.sql_args))
        if not obs:
            return None
        return obs[0]