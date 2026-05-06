def transform(self, jam, query=None):
        '''Transform jam object to make data for this task

        Parameters
        ----------
        jam : jams.JAMS
            The jams container object

        query : string, dict, or callable [optional]
            An optional query to narrow the elements of `jam.annotations`
            to be considered.

            If not provided, all annotations are considered.

        Returns
        -------
        data : dict
            A dictionary of transformed annotations.
            All annotations which can be converted to the target namespace
            will be converted.
        '''
        anns = []
        if query:
            results = jam.search(**query)
        else:
            results = jam.annotations

        # Find annotations that can be coerced to our target namespace
        for ann in results:
            try:
                anns.append(jams.nsconvert.convert(ann, self.namespace))
            except jams.NamespaceError:
                pass

        duration = jam.file_metadata.duration

        # If none, make a fake one
        if not anns:
            anns = [self.empty(duration)]

        # Apply transformations
        results = []
        for ann in anns:

            results.append(self.transform_annotation(ann, duration))
            # If the annotation range is None, it spans the entire track
            if ann.time is None or ann.duration is None:
                valid = [0, duration]
            else:
                valid = [ann.time, ann.time + ann.duration]

            results[-1]['_valid'] = time_to_frames(valid, sr=self.sr,
                                                   hop_length=self.hop_length)

        # Prefix and collect
        return self.merge(results)