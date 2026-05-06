def set_asset_dir(self):
        """Returns the ASSET_DIR environment variable

        This method gets the ASSET_DIR environment variable for the
        current asset install. It returns either the string value if
        set or None if it is not set.

        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.get_asset_dir')
        try:
            self.asset_dir = os.environ['ASSET_DIR']
        except KeyError:
            log.warn('Environment variable ASSET_DIR is not set!')
        else:
            log.info('Found environment variable ASSET_DIR: {a}'.format(a=self.asset_dir))