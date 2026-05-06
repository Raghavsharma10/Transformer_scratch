def _load(self):
        """Function to collect reference data and connect it to the instance as
         attributes.

         Internal function, does not usually need to be called by the user, as
         it is called automatically when an attribute is requested.

        :return None
        """

        data = get_data(self.endpoint, self.id_, force_lookup=self.__force_lookup)

        # Make our custom objects from the data.
        for key, val in data.items():

            if key == 'location_area_encounters' \
                    and self.endpoint == 'pokemon':

                params = val.split('/')[-3:]
                ep, id_, subr = params
                encounters = get_data(ep, int(id_), subr)
                data[key] = [_make_obj(enc) for enc in encounters]
                continue

            if isinstance(val, dict):
                data[key] = _make_obj(val)

            elif isinstance(val, list):
                data[key] = [_make_obj(i) for i in val]

        self.__dict__.update(data)

        return None