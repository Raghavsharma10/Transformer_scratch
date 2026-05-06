def create_from_json(cls, json_data):
        """Deserialize msa json data into a Msa object

        Args:
            json_data (dict): The json data for this msa

        Returns:
            Msa object

        """
        msa = Msa()
        msa.msa = json_data["msa_info"]["msa"]
        msa.meta = json_data["meta"] if "meta" in json_data else None

        msa.component_results = _create_component_results(json_data, "msa_info")

        return msa