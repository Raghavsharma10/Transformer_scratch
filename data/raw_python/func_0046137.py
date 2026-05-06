def get_provider_id(self):
        """Gets the ``Id`` of the provider.

        return: (osid.id.Id) - the provider ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        if ('providerId' not in self.my_osid_object._my_map or
                not self.my_osid_object._my_map['providerId']):
            raise IllegalState('this sourceable object has no provider set')
        return Id(self.my_osid_object._my_map['providerId'])