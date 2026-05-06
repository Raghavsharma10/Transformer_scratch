def load_locations(self, location_file=None):
        """Load locations into this resolver from the given
        *location_file*, which should contain one JSON object per line
        representing a location.  If *location_file* is not specified,
        an internal location database is used."""
        if location_file is None:
            contents = pkgutil.get_data(__package__, 'data/locations.json')
            contents_string = contents.decode("ascii")
            locations = contents_string.split('\n')
        else:
            from .cli import open_file
            with open_file(location_file, 'rb') as input:
                locations = input.readlines()
        
        for location_string in locations:
            if location_string.strip():
                location = Location(known=True, **json.loads(location_string))
                self.location_id_to_location[location.id] = location
                self.add_location(location)