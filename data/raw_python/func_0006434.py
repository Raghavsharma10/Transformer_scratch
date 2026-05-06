def save_configuration(self, configuration_file):
        '''Saving configuration

        Parameters
        ----------
        configuration_file : string
            Filename of the configuration file.
        '''
        if not isinstance(configuration_file, tb.file.File) and os.path.splitext(configuration_file)[1].strip().lower() != ".h5":
            return save_configuration_to_text_file(self, configuration_file)
        else:
            return save_configuration_to_hdf5(self, configuration_file)