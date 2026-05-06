def set_brightness(self, brightness):
        """Set dimmer brightness.

        Converts the Vera level property for dimmable lights from a percentage
        to the 0 - 255 scale used by HA.
        """
        percent = 0
        if brightness > 0:
            percent = round(brightness / 2.55)

        self.set_service_value(
            self.dimmer_service,
            'LoadLevelTarget',
            'newLoadlevelTarget',
            percent)
        self.set_cache_value('level', percent)