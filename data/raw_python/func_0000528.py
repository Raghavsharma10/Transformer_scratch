def from_dict(raw_data):
        """Create Term from raw dictionary data."""
        try:
            definition = raw_data['definition']
            term_id = raw_data['id']
            image = Image.from_dict(raw_data['image'])
            rank = raw_data['rank']
            term = raw_data['term']
            return Term(definition, term_id, image, rank, term)
        except KeyError:
            raise ValueError('Unexpected term json structure')