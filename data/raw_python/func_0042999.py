def register_observation(self, obstory_name, user_id, obs_time, obs_type, obs_meta=None):
        """
        Register a new observation, updating the database and returning the corresponding Observation object

        :param string obstory_name:
            The ID of the obstory which produced this observation
        :param string user_id:
            The ID of the user who created this observation
        :param float obs_time:
            The UTC date/time of the observation
        :param string obs_type:
            A string describing the semantic type of this observation
        :param list obs_meta:
            A list of :class:`meteorpi_model.Meta` used to provide additional information about this observation
        :return:
            The :class:`meteorpi_model.Observation` as stored in the database
        """

        if obs_meta is None:
            obs_meta = []

        # Get obstory id from name
        obstory = self.get_obstory_from_name(obstory_name)

        # Create a unique ID for this observation
        observation_id = mp.get_hash(obs_time, obstory['publicId'], obs_type)

        # Get ID code for obs_type
        obs_type_id = self.get_obs_type_id(obs_type)

        # Insert into database
        self.con.execute("""
INSERT INTO archive_observations (publicId, observatory, userId, obsTime, obsType)
VALUES
(%s, %s, %s, %s, %s);
""", (observation_id, obstory['uid'], user_id, obs_time, obs_type_id))

        # Store the observation metadata
        for meta in obs_meta:
            self.set_observation_metadata(user_id, observation_id, meta, obs_time)

        observation = mp.Observation(obstory_name=obstory_name,
                                     obstory_id=obstory['publicId'],
                                     obs_time=obs_time,
                                     obs_id=observation_id,
                                     obs_type=obs_type,
                                     file_records=[],
                                     meta=obs_meta)
        return observation