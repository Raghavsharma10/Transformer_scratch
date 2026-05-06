def query_single(self, object_type, url_params, query_params=None):
        # type: (str, list, dict) -> dict
        """
        Query for a single object.
        :param object_type: string query type (e.g., "users" or "groups")
        :param url_params: required list of strings to provide as additional URL components
        :param query_params: optional dictionary of query options
        :return: the found object (a dictionary), which is empty if none were found
        """
        # Server API convention (v2) is that the pluralized object type goes into the endpoint
        # but the object type is the key in the response dictionary for the returned object.
        self.local_status["single-query-count"] += 1
        query_type = object_type + "s"  # poor man's plural
        query_path = "/organizations/{}/{}".format(self.org_id, query_type)
        for component in url_params if url_params else []:
            query_path += "/" + urlparse.quote(component, safe='/@')
        if query_params: query_path += "?" + urlparse.urlencode(query_params)
        try:
            result = self.make_call(query_path)
            body = result.json()
        except RequestError as re:
            if re.result.status_code == 404:
                if self.logger: self.logger.debug("Ran %s query: %s %s (0 found)",
                                                  object_type, url_params, query_params)
                return {}
            else:
                raise re
        if body.get("result") == "success":
            value = body.get(object_type, {})
            if self.logger: self.logger.debug("Ran %s query: %s %s (1 found)", object_type, url_params, query_params)
            return value
        else:
            raise ClientError("OK status but no 'success' result", result)