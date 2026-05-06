def query_image_id(self, image_id):
        """Query OPUS via the image_id.

        This is a query using the 'primaryfilespec' field of the OPUS database.
        It returns a list of URLS into the `obsids` attribute.

        This example queries for an image of Titan:

        >>> opus = opusapi.OPUS()
        >>> opus.query_image_id('N1695760475_1')

        After this, one can call `download_results()` to retrieve the found
        data into the standard locations into the database_path as defined in
        `.pyciss.yaml` (the config file),
        """
        myquery = {"primaryfilespec": image_id}
        self.create_files_request(myquery, fmt="json")
        self.unpack_json_response()
        return self.obsids