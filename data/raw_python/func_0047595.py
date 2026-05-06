def _effective_view_filter(self):
        """Returns the mongodb relationship filter for effective views"""
        if self._effective_view == EFFECTIVE:
            now = datetime.datetime.utcnow()
            return {'startDate': {'$$lte': now}, 'endDate': {'$$gte': now}}
        return {}