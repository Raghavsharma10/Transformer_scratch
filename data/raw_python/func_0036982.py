def _handle_response(self, response, valid_status_codes, resource):
        """
        Handles Response objects

        Args:
            response: An HTTP reponse object
            valid_status_codes: A tuple list of valid status codes
            resource: The resource class to build from this response

        returns:
            resources: A list of Resource instances
        """
        if response.status_code not in valid_status_codes:
            raise InvalidStatusCodeError(
                status_code=response.status_code,
                expected_status_codes=valid_status_codes
                )
        if response.content:
            data = response.json()
            if isinstance(data, list):
                # A list of results is always rendered
                return [resource(**x) for x in data]
            else:
                # Try and find the paginated resources
                key = getattr(resource.Meta, 'pagination_key', None)
                if isinstance(data.get(key), list):
                    # Only return the paginated responses
                    return [resource(**x) for x in data.get(key)]
                else:
                    # Attempt to render this whole response as a resource
                    return [resource(**data)]
        return []