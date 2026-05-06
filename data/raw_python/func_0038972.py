def transform(self, func, func_description=None):
        """
        Applies a given a function to the features of each subject
            and returns a new dataset with other info unchanged.

        Parameters
        ----------
        func : callable
            A valid callable that takes in a single ndarray and returns a single ndarray.
            Ensure the transformed dimensionality must be the same for all subjects.

            If your function requires more than one argument,
            use `functools.partial` to freeze all the arguments
            except the features for the subject.

        func_description : str, optional
            Human readable description of the given function.

        Returns
        -------
        xfm_ds : MLDataset
            with features obtained from subject-wise transform

        Raises
        ------
        TypeError
            If given func is not a callable
        ValueError
            If transformation of any of the subjects features raises an exception.

        Examples
        --------
        Simple:

        .. code-block:: python

            from pyradigm import MLDataset

            thickness = MLDataset(in_path='ADNI_thickness.csv')
            pcg_thickness = thickness.apply_xfm(func=get_pcg, description = 'applying ROI mask for PCG')
            pcg_median = pcg_thickness.apply_xfm(func=np.median, description='median per subject')


        Complex example with function taking more than one argument:

        .. code-block:: python

            from pyradigm import MLDataset
            from functools import partial
            import hiwenet

            thickness = MLDataset(in_path='ADNI_thickness.csv')
            roi_membership = read_roi_membership()
            hw = partial(hiwenet, groups = roi_membership)

            thickness_hiwenet = thickness.transform(func=hw, description = 'histogram weighted networks')
            median_thk_hiwenet = thickness_hiwenet.transform(func=np.median, description='median per subject')

        """

        if not callable(func):
            raise TypeError('Given function {} is not a callable'.format(func))

        xfm_ds = MLDataset()
        for sample, data in self.__data.items():
            try:
                xfm_data = func(data)
            except:
                print('Unable to transform features for {}. Quitting.'.format(sample))
                raise

            xfm_ds.add_sample(sample, xfm_data,
                              label=self.__labels[sample],
                              class_id=self.__classes[sample])

        xfm_ds.description = "{}\n{}".format(func_description, self.__description)

        return xfm_ds