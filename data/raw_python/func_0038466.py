def delete(self, resource_path: str, administration_id: int = None):
        """
        Performs a DELETE request to the endpoint identified by the resource path. DELETE requests are usually used to
        (permanently) delete existing data. USE THIS METHOD WITH CAUTION.

        From a client perspective, DELETE requests behave similarly to GET requests.

        :param resource_path: The resource path.
        :param administration_id: The administration id (optional, depending on the resource path).
        :return: The decoded JSON response for the request.
        """
        response = self.session.delete(
            url=self._get_url(administration_id, resource_path),
        )
        return self._process_response(response)