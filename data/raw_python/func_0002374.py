def _get_keywords(self, location, keywords):
        """Format GET request's parameters from keywords."""
        if 'xml' in keywords:
            keywords.pop('xml')
            self.xml = True
        else:
            keywords['file_type'] = 'json'
        if 'id' in keywords:
            if location != 'series':
                location = location.rstrip('s')
            key = '%s_id' % location
            value = keywords.pop('id')
            keywords[key] = value
        if 'start' in keywords:
            time = keywords.pop('start')
            keywords['realtime_start'] = time
        if 'end' in keywords:
            time = keywords.pop('end')
            keywords['realtime_end'] = time
        if 'sort' in keywords:
            order = keywords.pop('sort')
            keywords['sort_order'] = order
        keywords['api_key'] = self.api_key
        return keywords