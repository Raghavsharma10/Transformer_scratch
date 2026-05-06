def _update_guide(self, guide, update=False, clear=True):
        """Update a single specified guide"""

        kml_filename = os.path.join(self.cache_path, guide + '.kml')
        geojson_filename = os.path.join(self.cache_path, guide + '.geojson')

        if not os.path.exists(geojson_filename) or update:
            try:
                data = request.urlopen(self.guides[guide]).read().decode(
                    'utf-8')
            except (request.URLError, request.HTTPError) as e:
                self.log('Could not get web guide data:', e, type(e), lvl=warn)
                return

            with open(kml_filename, 'w') as f:
                f.write(data)

            self._translate(kml_filename, geojson_filename)

        with open(geojson_filename, 'r') as f:
            json_data = json.loads(f.read())

        if len(json_data['features']) == 0:
            self.log('No features found!', lvl=warn)
            return

        layer = objectmodels['layer'].find_one({'name': guide})

        if clear and layer is not None:
            layer.delete()
            layer = None

        if layer is None:
            layer_uuid = std_uuid()
            layer = objectmodels['layer']({
                'uuid': layer_uuid,
                'name': guide,
                'type': 'geoobjects'
            })
            layer.save()
        else:
            layer_uuid = layer.uuid

        if clear:
            for item in objectmodels['geoobject'].find({'layer': layer_uuid}):
                self.log('Deleting old guide location', lvl=debug)
                item.delete()

        locations = []

        for item in json_data['features']:
            self.log('Adding new guide location:', item, lvl=verbose)
            location = objectmodels['geoobject']({
                'uuid': std_uuid(),
                'layer': layer_uuid,
                'geojson': item,
                'type': 'Skipperguide',
                'name': 'Guide for %s' % (item['properties']['Name'])
            })
            locations.append(location)

        self.log('Bulk inserting guide locations', lvl=debug)
        objectmodels['geoobject'].bulk_create(locations)