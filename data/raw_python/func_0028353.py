def create_from_json(cls, json_data):
        """Deserialize block json data into a Block object

        Args:
            json_data (dict): The json data for this block

        Returns:
            Block object

        """
        block = Block()
        block_info = json_data["block_info"]
        block.block_id = block_info["block_id"]
        block.num_bins = block_info["num_bins"] if "num_bins" in block_info else None
        block.property_type = block_info["property_type"] if "property_type" in block_info else None
        block.meta = json_data["meta"] if "meta" in json_data else None

        block.component_results = _create_component_results(json_data, "block_info")

        return block