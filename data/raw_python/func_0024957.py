def create_manifest_from_space(self):
        """
        Populate a manifest file generated from details from the
        cloud foundry space environment.
        """
        space = predix.admin.cf.spaces.Space()

        summary = space.get_space_summary()
        for instance in summary['services']:
            service_type = instance['service_plan']['service']['label']
            name = instance['name']
            if service_type in self.supported:
                service = self.supported[service_type](name=name)
                service.add_to_manifest(self)
            elif service_type == 'us-weather-forecast':
                weather = predix.admin.weather.WeatherForecast(name=name)
                weather.add_to_manifest(self)
            else:
                logging.warning("Unsupported service type: %s" % service_type)