def map_services(self):
        """Get full Vera device service info."""
        # the Vera rest API is a bit rough so we need to make 2 calls
        # to get all the info e need
        self.get_simple_devices_info()

        j = self.data_request({'id': 'status', 'output_format': 'json'}).json()

        service_map = {}

        items = j.get('devices')

        for item in items:
            service_map[item.get('id')] = item.get('states')

        self.device_services_map = service_map