def sensor_values(self):
        """
        Returns the values of all sensors for this cluster
        """
        self.update_instance_sensors(opt="all")
        return {
            "light": self.lux,
            "water": self.soil_moisture,
            "humidity": self.humidity,
            "temperature": self.temp
        }