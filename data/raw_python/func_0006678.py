def write_json(metadata):
        """
        Write the metadata object to file
        :param metadata: Metadata object
        """
        # Open the metadata file to write
        with open(metadata.jsonfile, 'w') as metadatafile:
            # Write the json dump of the object dump to the metadata file
            json.dump(metadata.dump(), metadatafile, sort_keys=True, indent=4, separators=(',', ': '))