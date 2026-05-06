def get_space_guid(self):
        """
        Returns the GUID for the space currently targeted.

        Can be set by environment variable with PREDIX_SPACE_GUID.
        Can be determined by ~/.cf/config.json.
        """
        if 'PREDIX_SPACE_GUID' in os.environ:
            return os.environ['PREDIX_SPACE_GUID']
        else:
            info = self._get_space_info()
            for key in ('Guid', 'GUID'):
                if key in info.keys():
                    return info[key]
            raise ValueError('Unable to determine cf space guid')