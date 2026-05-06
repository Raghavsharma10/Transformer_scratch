def get_pipeline(self, *args, **kwargs):
        '''
        Returns the `time` and `flux` arrays for the target obtained by a given
        pipeline.

        Options :py:obj:`args` and :py:obj:`kwargs` are passed directly to
        the :py:func:`pipelines.get` function of the mission.

        '''

        return getattr(missions, self.mission).pipelines.get(self.ID, *args,
                                                             **kwargs)