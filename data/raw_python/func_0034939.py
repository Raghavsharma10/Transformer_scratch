def hdf5_read_out(self):
        """Read out an hdf5 file.

        Takes the output of :class:`gwsnrcalc.genconutils.genprocess.GenProcess`
        and reads it out to an HDF5 file.

        """
        with h5py.File(self.WORKING_DIRECTORY + '/' + self.output_file_name, 'w') as f:

            header = f.create_group('header')
            header.attrs['Title'] = 'Generated SNR Out'
            header.attrs['Author'] = 'Generator by: Michael Katz'
            header.attrs['Date/Time'] = str(datetime.datetime.now())

            for which in ['x', 'y']:
                header.attrs[which + 'val_name'] = getattr(self, which + 'val_name')
                header.attrs['num_' + which + '_pts'] = getattr(self, 'num_' + which)

            ecc = 'eccentricity' in self.__dict__
            if ecc:
                name_list = ['observation_time', 'start_frequency', 'start_separation'
                             'eccentricity']
            else:
                name_list = ['spin_1', 'spin_2', 'spin', 'end_time']

            name_list += ['total_mass', 'mass_ratio', 'start_time', 'luminosity_distance',
                          'comoving_distance', 'redshift']

            for name in name_list:
                if name != self.xval_name and name != self.yval_name:
                    try:
                        getattr(self, name)
                        header.attrs[name] = getattr(self, name)
                    except AttributeError:
                        pass

            if self.added_note != '':
                header.attrs['Added note'] = self.added_note

            data = f.create_group('data')

            # read out x,y values in compressed data set
            dset = data.create_dataset(self.x_col_name, data=self.xvals,
                                       dtype='float64', chunks=True,
                                       compression='gzip', compression_opts=9)

            dset = data.create_dataset(self.y_col_name, data=self.yvals,
                                       dtype='float64', chunks=True,
                                       compression='gzip', compression_opts=9)

            # read out all datasets
            for key in self.output_dict.keys():
                dset = data.create_dataset(key, data=self.output_dict[key],
                                           dtype='float64', chunks=True,
                                           compression='gzip', compression_opts=9)