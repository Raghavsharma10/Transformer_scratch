def get_collection(cls, request_args):
        r"""
        Used to fetch a collection of resource object of type 'cls' in response to a GET request\
        . get_resource_or_collection should only be invoked on a resource when the client specifies a GET request.

        :param request_args: The query parameters supplied with the request.  currently supports page[offset], and \
        page[limit]. Pagination only applies to collection requests. See http://jsonapi.org/format/#fetching-pagination.
        :return: An HTTP response object in accordance with the specification at \
        http://jsonapi.org/format/#fetching-resources
        """
        try:
            if request_args.get('include'):
                raise ParameterNotSupported

            offset = request_args.get('page[offset]', 0)
            limit = request_args.get('page[limit]', 20)

            query = "MATCH (n) WHERE n:{label} AND n.active RETURN n ORDER BY n.id SKIP {offset} LIMIT {limit}".format(
                label=cls.__name__,
                offset=offset,
                limit=limit)

            results, meta = db.cypher_query(query)
            data = dict()
            data['data'] = list()
            data['links'] = dict()

            data['links']['self'] = "{class_link}?page[offset]={offset}&page[limit]={limit}".format(
                class_link=cls.get_class_link(),
                offset=offset,
                limit=limit
            )

            data['links']['first'] = "{class_link}?page[offset]={offset}&page[limit]={limit}".format(
                class_link=cls.get_class_link(),
                offset=0,
                limit=limit
            )

            if int(offset) - int(limit) > 0:
                data['links']['prev'] = "{class_link}?page[offset]={offset}&page[limit]={limit}".format(
                    class_link=cls.get_class_link(),
                    offset=int(offset)-int(limit),
                    limit=limit
                )

            if len(cls.nodes) > int(offset) + int(limit):
                data['links']['next'] = "{class_link}?page[offset]={offset}&page[limit]={limit}".format(
                    class_link=cls.get_class_link(),
                    offset=int(offset)+int(limit),
                    limit=limit
                )

            data['links']['last'] = "{class_link}?page[offset]={offset}&page[limit]={limit}".format(
                class_link=cls.get_class_link(),
                offset=len(cls.nodes.filter(active=True)) - (len(cls.nodes.filter(active=True)) % int(limit))-1,
                limit=limit
            )

            list_of_nodes = [cls.inflate(row[0]) for row in results]
            for this_node in list_of_nodes:
                data['data'].append(this_node.get_resource_object())
            r = make_response(jsonify(data))
            r.status_code = http_error_codes.OK
            r.headers['Content-Type'] = CONTENT_TYPE
            return r
        except ParameterNotSupported:
            return application_codes.error_response([application_codes.PARAMETER_NOT_SUPPORTED_VIOLATION])