def _get_common_kwargs(self, system):
        """ Get kwargs common for all methods. """
        enc_class = getattr(system['view'], '_json_encoder', None)
        if enc_class is None:
            enc_class = get_json_encoder()
        return {
            'request': system['request'],
            'encoder': enc_class,
        }