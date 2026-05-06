def to_dict(self):
        """
        Return a dict representation of an osmnet osmnet_config instance.
        """
        return {'logs_folder': self.logs_folder,
                'log_file': self.log_file,
                'log_console': self.log_console,
                'log_name': self.log_name,
                'log_filename': self.log_filename,
                'keep_osm_tags': self.keep_osm_tags
                }