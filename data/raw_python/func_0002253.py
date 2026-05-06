def set_lock_config(self, device_label, volume=None, voice_level=None,
                        auto_lock_enabled=None):
        """ Set lock configuration

        Args:
            device_label (str): device label of lock
            volume (str): 'SILENCE', 'LOW' or 'HIGH'
            voice_level (str): 'ESSENTIAL' or 'NORMAL'
            auto_lock_enabled (boolean): auto lock enabled
        """
        response = None
        data = {}
        if volume:
            data['volume'] = volume
        if voice_level:
            data['voiceLevel'] = voice_level
        if auto_lock_enabled is not None:
            data['autoLockEnabled'] = auto_lock_enabled
        try:
            response = requests.put(
                urls.lockconfig(self._giid, device_label),
                headers={
                    'Content-Type': 'application/json',
                    'Cookie': 'vid={}'.format(self._vid)},
                data=json.dumps(data))
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)