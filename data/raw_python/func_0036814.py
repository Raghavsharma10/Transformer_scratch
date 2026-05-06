def search(self):
        r"""
            Call the Bugzilla endpoint that will do the search. It will take
            the information used in other methods on the Search object and
            build up the query string. If no bugs are found then an empty list
            is returned.

            >>> bugs = bugzilla.search_for\
            ...                .keywords("checkin-needed")\
            ...                .include_fields("flags")\
            ...                .search()
        """
        params = {}
        params.update(self._time_frame.items())

        if self._includefields:
            params['include_fields'] = list(self._includefields)
        if self._bug_numbers:
            bugs = []
            for bug in self._bug_numbers:
                result = self._bugsy.request('bug/%s' % bug,
                                             params=params)
                bugs.append(Bug(self._bugsy, **result['bugs'][0]))

            return bugs
        else:
            if self._component:
                params['component'] = list(self._component)
            if self._product:
                params['product'] = list(self._product)
            if self._keywords:
                params['keywords'] = list(self._keywords)
            if self._assigned:
                params['assigned_to'] = list(self._assigned)
            if self._summaries:
                params['short_desc_type'] = 'allwordssubstr'
                params['short_desc'] = list(self._summaries)
            if self._whiteboard:
                params['short_desc_type'] = 'allwordssubstr'
                params['whiteboard'] = list(self._whiteboard)
            if self._change_history['fields']:
                params['chfield'] = self._change_history['fields']
            if self._change_history.get('value', None):
                params['chfieldvalue'] = self._change_history['value']

            try:
                results = self._bugsy.request('bug', params=params)
            except Exception as e:
                raise SearchException(e.msg, e.code)

            return [Bug(self._bugsy, **bug) for bug in results['bugs']]