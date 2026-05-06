def transform(self, audio_f=None, jam=None, y=None, sr=None, crop=False):
        '''Apply the transformations to an audio file, and optionally JAMS object.

        Parameters
        ----------
        audio_f : str
            Path to audio file

        jam : optional, `jams.JAMS`, str or file-like
            Optional JAMS object/path to JAMS file/open file descriptor.

            If provided, this will provide data for task transformers.

        y : np.ndarray
        sr : number > 0
            If provided, operate directly on an existing audio buffer `y` at
            sampling rate `sr` rather than load from `audio_f`.

        crop : bool
            If `True`, then data are cropped to a common time index across all
            fields.  Otherwise, data may have different time extents.

        Returns
        -------
        data : dict
            Data dictionary containing the transformed audio (and annotations)

        Raises
        ------
        ParameterError
            At least one of `audio_f` or `(y, sr)` must be provided.

        '''

        if y is None:
            if audio_f is None:
                raise ParameterError('At least one of `y` or `audio_f` '
                                     'must be provided')

            # Load the audio
            y, sr = librosa.load(audio_f, sr=sr, mono=True)

        if sr is None:
            raise ParameterError('If audio is provided as `y`, you must '
                                 'specify the sampling rate as sr=')

        if jam is None:
            jam = jams.JAMS()
            jam.file_metadata.duration = librosa.get_duration(y=y, sr=sr)

        # Load the jams
        if not isinstance(jam, jams.JAMS):
            jam = jams.load(jam)

        data = dict()

        for operator in self.ops:
            if isinstance(operator, BaseTaskTransformer):
                data.update(operator.transform(jam))
            elif isinstance(operator, FeatureExtractor):
                data.update(operator.transform(y, sr))
        if crop:
            data = self.crop(data)
        return data