def txt_read_out(self):
        """Read out txt file.

        Takes the output of :class:`gwsnrcalc.genconutils.genprocess.GenProcess`
        and reads it out to a txt file.

        """

        header = '#Generated SNR Out\n'
        header += '#Generator by: Michael Katz\n'
        header += '#Date/Time: {}\n'.format(datetime.datetime.now())

        for which in ['x', 'y']:
            header += '#' + which + 'val_name: {}\n'.format(getattr(self, which + 'val_name'))
            header += '#num_' + which + '_pts: {}\n'.format(getattr(self, 'num_' + which))

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
                    header += '#{}: {}\n'.format(name, getattr(self, name))
                except AttributeError:
                    pass

        if self.added_note != '':
            header += '#Added note: ' + self.added_note + '\n'
        else:
            header += '#Added note: None\n'

        header += '#--------------------\n'

        header += self.x_col_name + '\t'

        header += self.y_col_name + '\t'

        for key in self.output_dict.keys():
            header += key + '\t'

        # read out x,y and the data
        x_and_y = np.asarray([self.xvals, self.yvals])
        snr_out = np.asarray([self.output_dict[key] for key in self.output_dict.keys()]).T

        data_out = np.concatenate([x_and_y.T, snr_out], axis=1)

        np.savetxt(self.WORKING_DIRECTORY + '/' + self.output_file_name,
                   data_out, delimiter='\t', header=header, comments='')
        return