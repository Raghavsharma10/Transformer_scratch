def get_xyz(self, list_of_names=None):
        """Get xyz coordinates for these electrodes

        Parameters
        ----------
        list_of_names : list of str
            list of electrode names to use

        Returns
        -------
        list of tuples of 3 floats (x, y, z)
            list of xyz coordinates for all the electrodes

        TODO
        ----
        coordinate system of electrodes
        """
        if list_of_names is not None:
            filter_lambda = lambda x: x['name'] in list_of_names
        else:
            filter_lambda = None

        return self.electrodes.get(filter_lambda=filter_lambda,
                                   map_lambda=lambda e: (float(e['x']),
                                                         float(e['y']),
                                                         float(e['z'])))