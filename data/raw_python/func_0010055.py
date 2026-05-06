def get_data_path(self, filename, env_prefix=None):
        """
        Get data path.

        Args:
            filename (string) : Name of file inside of /data folder to retrieve.

        Kwargs:
            env_prefix (string) : Name of subfolder, ex: 'qa' will find files in /data/qa

        Returns:
            String - path to file.

        Usage::

            open(WTF_DATA_MANAGER.get_data_path('testdata.csv')

        Note: WTF_DATA_MANAGER is a provided global instance of DataManager

        """
        if env_prefix == None:
            target_file = filename
        else:
            target_file = os.path.join(env_prefix, filename)

        if os.path.exists(os.path.join(self._data_path, target_file)):
            return os.path.join(self._data_path, target_file)
        else:
            raise DataNotFoundError(
                u("Cannot find data file: {0}").format(target_file))