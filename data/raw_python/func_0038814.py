async def dispatch(self, request, view=None, **kwargs):
        """Process request."""
        # Authorization endpoint
        self.auth = await self.authorize(request, **kwargs)  # noqa

        # Load collection
        self.collection = await self.get_many(request, **kwargs)

        if request.method == 'POST' and view is None:
            return await super(RESTHandler, self).dispatch(request, **kwargs)

        # Load resource
        resource = await self.get_one(request, **kwargs)

        headers = {}

        if request.method == 'GET' and resource is None:

            # Filter resources
            if VAR_WHERE in request.query:
                self.collection = await self.filter(request, **kwargs)

            # Sort resources
            if VAR_SORT in request.query:
                sorting = [(name.strip('-'), name.startswith('-'))
                           for name in request.query[VAR_SORT].split(',')]
                self.collection = await self.sort(*sorting, **kwargs)

            # Paginate resources
            per_page = request.query.get(VAR_PER_PAGE, self.meta.per_page)
            if per_page:
                try:
                    per_page = int(per_page)
                    if per_page:
                        page = int(request.query.get(VAR_PAGE, 0))
                        offset = page * per_page
                        self.collection, total = await self.paginate(request, offset, per_page)
                        headers = make_pagination_headers(
                            request, per_page, page, total, self.meta.page_links)
                except ValueError:
                    raise RESTBadRequest(reason='Pagination params are invalid.')

        response = await super(RESTHandler, self).dispatch(
            request, resource=resource, view=view, **kwargs)
        response.headers.update(headers)
        return response