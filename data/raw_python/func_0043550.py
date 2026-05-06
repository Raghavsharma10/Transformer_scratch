def factory(data):
        """
        Try to reconstruct the APIResource from its data.

        :param data: The APIResource data
        :type data: dict

        :return: The guessed APIResource

        :raise
            exceptions.UnkownAPIResource when it's impossible to reconstruct the APIResource from its data.
        """
        if 'object' not in data:
            raise exceptions.UnknownAPIResource('Missing `object` key in resource.')

        for reconstituable_api_resource_type in ReconstituableAPIResource.__subclasses__():
            if reconstituable_api_resource_type.object_type == data['object']:
                return reconstituable_api_resource_type(**data)

        raise exceptions.UnknownAPIResource('Unknown object `' + data['object'] + '`.')