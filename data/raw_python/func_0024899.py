def get_organization_guid(self):
        """
        Returns the GUID for the organization currently targeted.
        """
        if 'PREDIX_ORGANIZATION_GUID' in os.environ:
            return os.environ['PREDIX_ORGANIZATION_GUID']
        else:
            info = self._get_organization_info()
            for key in ('Guid', 'GUID'):
                if key in info.keys():
                    return info[key]
            raise ValueError('Unable to determine cf organization guid')