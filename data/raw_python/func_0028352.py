def create_from_json(cls, json_data):
        """Deserialize property json data into a Property object

        Args:
            json_data (dict): The json data for this property

        Returns:
            Property object

        """
        prop = Property()
        address_info = json_data["address_info"]
        prop.address = address_info["address"]
        prop.block_id = address_info["block_id"]
        prop.zipcode = address_info["zipcode"]
        prop.zipcode_plus4 = address_info["zipcode_plus4"]
        prop.address_full = address_info["address_full"]
        prop.city = address_info["city"]
        prop.county_fips = address_info["county_fips"]
        prop.geo_precision = address_info["geo_precision"]
        prop.lat = address_info["lat"]
        prop.lng = address_info["lng"]
        prop.slug = address_info["slug"]
        prop.state = address_info["state"]
        prop.unit = address_info["unit"]

        prop.meta = None
        if "meta" in json_data:
            prop.meta = json_data["meta"]

        prop.component_results = _create_component_results(json_data, "address_info")

        return prop