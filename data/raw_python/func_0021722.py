def _get_base_params(self):
        """Get the params that will be included with every request
        """
        base_params = {
            'locale':       self._get_locale(),
            'device_id':    ANDROID.DEVICE_ID,
            'device_type':  ANDROID.APP_PACKAGE,
            'access_token': ANDROID.ACCESS_TOKEN,
            'version':      ANDROID.APP_CODE,
        }
        base_params.update(dict((k, v) \
            for k, v in iteritems(self._state_params) \
                if v is not None))
        return base_params