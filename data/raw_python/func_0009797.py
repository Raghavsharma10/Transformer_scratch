def refresh_complex_value(self, name):
        """Refresh a value from the service dictionaries.

        It's best to use get_value / refresh if it has the data you need.
        """
        for item in self.json_state.get('states'):
            if item.get('variable') == name:
                service_id = item.get('service')
                result = self.vera_request(**{
                    'id': 'variableget',
                    'output_format': 'json',
                    'DeviceNum': self.device_id,
                    'serviceId': service_id,
                    'Variable': name
                })
                item['value'] = result.text
                return item.get('value')
        return None