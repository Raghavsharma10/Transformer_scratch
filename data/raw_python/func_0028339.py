def fetch(self, endpoint_name, identifier_input, query_params=None):
        """Calls this instance's request_client's post method with the
        specified component endpoint

        Args:
            - endpoint_name (str) - The endpoint to call like "property/value".
            - identifier_input - One or more identifiers to request data for. An identifier can
                be in one of these forms:

                - A list of property identifier dicts:
                    - A property identifier dict can contain the following keys:
                      (address, zipcode, unit, city, state, slug, meta).
                      One of 'address' or 'slug' is required.

                      Ex: [{"address": "82 County Line Rd",
                           "zipcode": "72173",
                           "meta": "some ID"}]

                      A slug is a URL-safe string that identifies a property.
                      These are obtained from HouseCanary.

                      Ex: [{"slug": "123-Example-St-San-Francisco-CA-94105"}]

                - A list of dicts representing a block:
                  - A block identifier dict can contain the following keys:
                      (block_id, num_bins, property_type, meta).
                      'block_id' is required.

                  Ex: [{"block_id": "060750615003005", "meta": "some ID"}]

                - A list of dicts representing a zipcode:

                  Ex: [{"zipcode": "90274", "meta": "some ID"}]

                - A list of dicts representing an MSA:

                  Ex: [{"msa": "41860", "meta": "some ID"}]

                The "meta" field is always optional.

        Returns:
            A Response object, or the output of a custom OutputGenerator
            if one was specified in the constructor.
        """

        endpoint_url = constants.URL_PREFIX + "/" + self._version + "/" + endpoint_name

        if query_params is None:
            query_params = {}

        if len(identifier_input) == 1:
            # If only one identifier specified, use a GET request
            query_params.update(identifier_input[0])
            return self._request_client.get(endpoint_url, query_params)

        # when more than one address, use a POST request
        return self._request_client.post(endpoint_url, identifier_input, query_params)