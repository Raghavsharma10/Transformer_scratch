def _prepare_headers(self, request, filter=None, order_by=None, group_by=[], page=None, page_size=None):
        """ Prepare headers for the given request

            Args:
                request: the NURESTRequest to send
                filter: string
                order_by: string
                group_by: list of names
                page: int
                page_size: int
        """

        if filter:
            request.set_header('X-Nuage-Filter', filter)

        if order_by:
            request.set_header('X-Nuage-OrderBy', order_by)

        if page is not None:
            request.set_header('X-Nuage-Page', str(page))

        if page_size:
            request.set_header('X-Nuage-PageSize', str(page_size))

        if len(group_by) > 0:
            header = ", ".join(group_by)
            request.set_header('X-Nuage-GroupBy', 'true')
            request.set_header('X-Nuage-Attributes', header)