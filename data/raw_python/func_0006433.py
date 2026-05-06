def load_configuration(self, configuration_file):
        '''Loading configuration

        Parameters
        ----------
        configuration_file : string
            Path to the configuration file (text or HDF5 file).
        '''
        if os.path.isfile(configuration_file):
            if not isinstance(configuration_file, tb.file.File) and os.path.splitext(configuration_file)[1].strip().lower() != ".h5":
                load_configuration_from_text_file(self, configuration_file)
            else:
                load_configuration_from_hdf5(self, configuration_file)
        else:
            raise ValueError('Cannot find configuration file specified: %s' % configuration_file)