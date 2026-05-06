def get_simple_devices_info(self):
        """Get basic device info from Vera."""
        j = self.data_request({'id': 'sdata'}).json()

        self.scenes = []
        items = j.get('scenes')

        for item in items:
            self.scenes.append(VeraScene(item, self))

        if j.get('temperature'):
            self.temperature_units = j.get('temperature')

        self.categories = {}
        cats = j.get('categories')

        for cat in cats:
            self.categories[cat.get('id')] = cat.get('name')

        self.device_id_map = {}

        devs = j.get('devices')
        for dev in devs:
            dev['categoryName'] = self.categories.get(dev.get('category'))
            self.device_id_map[dev.get('id')] = dev