def create_from_json(cls, json_data):
        """Deserialize zipcode json data into a ZipCode object

        Args:
            json_data (dict): The json data for this zipcode

        Returns:
            Zip object

        """
        zipcode = ZipCode()
        zipcode.zipcode = json_data["zipcode_info"]["zipcode"]
        zipcode.meta = json_data["meta"] if "meta" in json_data else None

        zipcode.component_results = _create_component_results(json_data, "zipcode_info")

        return zipcode