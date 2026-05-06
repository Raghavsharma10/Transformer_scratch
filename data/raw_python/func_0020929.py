def get_records(**kwargs):
    """Get records paginated."""
    page_ = kwargs.get('resumptionToken', {}).get('page', 1)
    size_ = current_app.config['OAISERVER_PAGE_SIZE']
    scroll = current_app.config['OAISERVER_RESUMPTION_TOKEN_EXPIRE_TIME']
    scroll_id = kwargs.get('resumptionToken', {}).get('scroll_id')

    if scroll_id is None:
        search = OAIServerSearch(
            index=current_app.config['OAISERVER_RECORD_INDEX'],
        ).params(
            scroll='{0}s'.format(scroll),
        ).extra(
            version=True,
        )[(page_-1)*size_:page_*size_]

        if 'set' in kwargs:
            search = search.query('match', **{'_oai.sets': kwargs['set']})

        time_range = {}
        if 'from_' in kwargs:
            time_range['gte'] = kwargs['from_']
        if 'until' in kwargs:
            time_range['lte'] = kwargs['until']
        if time_range:
            search = search.filter('range', **{'_updated': time_range})

        response = search.execute().to_dict()
    else:
        response = current_search_client.scroll(
            scroll_id=scroll_id,
            scroll='{0}s'.format(scroll),
        )

    class Pagination(object):
        """Dummy pagination class."""

        page = page_
        per_page = size_

        def __init__(self, response):
            """Initilize pagination."""
            self.response = response
            self.total = response['hits']['total']
            self._scroll_id = response.get('_scroll_id')

            # clean descriptor on last page
            if not self.has_next:
                current_search_client.clear_scroll(
                    scroll_id=self._scroll_id
                )
                self._scroll_id = None

        @cached_property
        def has_next(self):
            """Return True if there is next page."""
            return self.page * self.per_page <= self.total

        @cached_property
        def next_num(self):
            """Return next page number."""
            return self.page + 1 if self.has_next else None

        @property
        def items(self):
            """Return iterator."""
            from datetime import datetime
            for result in self.response['hits']['hits']:
                if '_oai' in result['_source']:
                    yield {
                        'id': result['_id'],
                        'json': result,
                        'updated': datetime.strptime(
                            result['_source']['_updated'][:19],
                            '%Y-%m-%dT%H:%M:%S'
                        ),
                    }

    return Pagination(response)