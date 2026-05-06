def from_dict(raw_data):
        """Create WordSet from raw dictionary data."""
        try:
            set_id = raw_data['id']
            title = raw_data['title']
            terms = [Term.from_dict(term) for term in raw_data['terms']]
            return WordSet(set_id, title, terms)
        except KeyError:
            raise ValueError('Unexpected set json structure')