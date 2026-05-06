def get_devices(self, category_filter=''):
        """Get list of connected devices.

        category_filter param is an array of strings
        """
        # pylint: disable=too-many-branches

        # the Vera rest API is a bit rough so we need to make 2 calls to get
        # all the info e need
        self.get_simple_devices_info()

        j = self.data_request({'id': 'status', 'output_format': 'json'}).json()

        self.devices = []
        items = j.get('devices')

        for item in items:
            item['deviceInfo'] = self.device_id_map.get(item.get('id'))
            if item.get('deviceInfo'):
                device_category = item.get('deviceInfo').get('category')
                if device_category == CATEGORY_DIMMER:
                    device = VeraDimmer(item, self)
                elif ( device_category == CATEGORY_SWITCH or
                       device_category == CATEGORY_VERA_SIREN):
                    device = VeraSwitch(item, self)
                elif device_category == CATEGORY_THERMOSTAT:
                    device = VeraThermostat(item, self)
                elif device_category == CATEGORY_LOCK:
                    device = VeraLock(item, self)
                elif device_category == CATEGORY_CURTAIN:
                    device = VeraCurtain(item, self)
                elif device_category == CATEGORY_ARMABLE:
                    device = VeraBinarySensor(item, self)
                elif (device_category == CATEGORY_SENSOR or
                      device_category == CATEGORY_HUMIDITY_SENSOR or
                      device_category == CATEGORY_TEMPERATURE_SENSOR or
                      device_category == CATEGORY_LIGHT_SENSOR or
                      device_category == CATEGORY_POWER_METER or
                      device_category == CATEGORY_UV_SENSOR):
                    device = VeraSensor(item, self)
                elif (device_category == CATEGORY_SCENE_CONTROLLER or
                      device_category == CATEGORY_REMOTE):
                    device = VeraSceneController(item, self)
                elif device_category == CATEGORY_GARAGE_DOOR:
                    device = VeraGarageDoor(item, self)
                else:
                    device = VeraDevice(item, self)
                self.devices.append(device)
                if (device.is_armable and not (
                    device_category == CATEGORY_SWITCH or
                    device_category == CATEGORY_VERA_SIREN or
                    device_category == CATEGORY_CURTAIN or
                    device_category == CATEGORY_GARAGE_DOOR)):
                    self.devices.append(VeraArmableDevice(item, self))
            else:
                self.devices.append(VeraDevice(item, self))

        if not category_filter:
            return self.devices

        devices = []
        for device in self.devices:
            if (device.category_name is not None and
                    device.category_name != '' and
                    device.category_name in category_filter):
                devices.append(device)
        return devices