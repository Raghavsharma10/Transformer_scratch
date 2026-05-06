def parse_xml_data(self):
        """
        Parses `xml_data` and loads it into object properties.
        """
        self.raw_text = self.xml_data.find('raw_text').text
        self.station = WeatherStation(self.xml_data.find('station_id').text)
        self.station.latitude = float(self.xml_data.find('latitude').text)
        self.station.longitude = float(self.xml_data.find('longitude').text)
        self.station.elevation = float(self.xml_data.find('elevation_m').text) * 3.28084