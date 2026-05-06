def query_multiple(self, object_type, page=0, url_params=None, query_params=None):
        # type: (str, int, list, dict) -> tuple
        """
        Query for a page of objects.  Defaults to the (0-based) first page.
        Sadly, the sort order is undetermined.
        :param object_type: string constant query type: either "user" or "group")
        :param page: numeric page (0-based) of results to get (up to 200 in a page)
        :param url_params: optional list of strings to provide as additional URL components
        :param query_params: optional dictionary of query options
        :return: tuple (list of returned dictionaries (one for each query result), bool for whether this is last page)
        """
        # As of 2017-10-01, we are moving to to different URLs for user and user-group queries,
        # and these endpoints have different conventions for pagination.  For the time being,
        # we are also preserving the more general "group" query capability.
        self.local_status["multiple-query-count"] += 1
        if object_type in ("user", "group"):
            query_path = "/{}s/{}/{:d}".format(object_type, self.org_id, page)
            if url_params: query_path += "/" + "/".join([urlparse.quote(c) for c in url_params])
            if query_params: query_path += "?" + urlparse.urlencode(query_params)
        elif object_type == "user-group":
            query_path = "/{}/user-groups".format(self.org_id)
            if url_params: query_path += "/" + "/".join([urlparse.quote(c) for c in url_params])
            query_path += "?page={:d}".format(page+1)
            if query_params: query_path += "&" + urlparse.urlencode(query_params)
        else:
            raise ArgumentError("Unknown query object type ({}): must be 'user' or 'group'".format(object_type))
        try:
            result = self.make_call(query_path)
            body = result.json()
        except RequestError as re:
            if re.result.status_code == 404:
                if self.logger: self.logger.debug("Ran %s query: %s %s (0 found)",
                                                  object_type, url_params, query_params)
                return [], True
            else:
                raise re
        if object_type in ("user", "group"):
            if body.get("result") == "success":
                values = body.get(object_type + "s", [])
                last_page = body.get("lastPage", False)
                if self.logger: self.logger.debug("Ran multi-%s query: %s %s (page %d: %d found)",
                                                  object_type, url_params, query_params, page, len(values))
                return values, last_page
            else:
                raise ClientError("OK status but no 'success' result", result)
        elif object_type == "user-group":
            page_number = result.headers.get("X-Current-Page", "1")
            page_count = result.headers.get("X-Page-Count", "1")
            if self.logger: self.logger.debug("Ran multi-group query: %s %s (page %d: %d found)",
                                              url_params, query_params, page, len(body))
            return body, int(page_number) >= int(page_count)
        else:
            # this would actually be caught above, but we use a parallel construction in both places
            # to make it easy to add query object types
            raise ArgumentError("Unknown query object type ({}): must be 'user' or 'group'".format(object_type))