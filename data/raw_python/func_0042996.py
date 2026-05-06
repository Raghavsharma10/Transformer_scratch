def has_observation_id(self, observation_id):
        """
        Check for the presence of the given observation_id

        :param string observation_id:
            The observation ID
        :return:
            True if we have a :class:`meteorpi_model.Observation` with this Id, False otherwise
        """
        self.con.execute('SELECT 1 FROM archive_observations WHERE publicId = %s', (observation_id,))
        return len(self.con.fetchall()) > 0