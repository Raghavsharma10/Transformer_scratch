def _translate(self, input_filename, output_filename):
        """Translate KML file to geojson for import"""
        command = [
            self.translate_binary,
            '-f', 'GeoJSON',
            output_filename,
            input_filename
        ]

        result = self._runcommand(command)
        self.log('Result (Translate): ', result, lvl=debug)