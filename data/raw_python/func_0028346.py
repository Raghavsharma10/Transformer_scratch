def component_mget(self, zip_data, components):
        """Call the zip component_mget endpoint

        Args:
            - zip_data - As described in the class docstring.
            - components - A list of strings for each component to include in the request.
                Example: ["zip/details", "zip/volatility"]
        """
        if not isinstance(components, list):
            print("Components param must be a list")
            return

        query_params = {"components": ",".join(components)}

        return self.fetch_identifier_component(
            "zip/component_mget", zip_data, query_params)