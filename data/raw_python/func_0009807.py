def set_color(self, rgb):
        """Set dimmer color.
        """

        target = ','.join([str(c) for c in rgb])
        self.set_service_value(
            self.color_service,
            'ColorRGB',
            'newColorRGBTarget',
            target)

        rgbi = self.get_color_index(['R', 'G', 'B'])
        if rgbi is None:
            return

        target = ('0=0,1=0,' +
                  str(rgbi[0]) + '=' + str(rgb[0]) + ',' +
                  str(rgbi[1]) + '=' + str(rgb[1]) + ',' +
                  str(rgbi[2]) + '=' + str(rgb[2]))
        self.set_cache_complex_value("CurrentColor", target)