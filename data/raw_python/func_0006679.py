def read_json(json_metadata):
        """
        Read the metadata object from file
        :param json_metadata: Path and file name of JSON-formatted metadata object file
        :return: metadata object
        """
        # Load the metadata object from the file
        with open(json_metadata) as metadatareport:
            jsondata = json.load(metadatareport)
        # Create the metadata objects
        metadata = MetadataObject()
        # Initialise the metadata categories as GenObjects created using the appropriate key
        for attr in jsondata:
            if not isinstance(jsondata[attr], dict):
                setattr(metadata, attr, jsondata[attr])
            else:
                setattr(metadata, attr, GenObject(jsondata[attr]))
        return metadata