def to_dict(self):
        """
        Convert current Pipeline (i.e. its attributes) into a dictionary

        :return: python dictionary
        """

        pipeline_desc_as_dict = {

            'uid': self._uid,
            'name': self._name,
            'state': self._state,
            'state_history': self._state_history,
            'completed': self._completed_flag.is_set()
        }

        return pipeline_desc_as_dict