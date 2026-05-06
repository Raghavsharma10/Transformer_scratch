def indexer_receiver(sender, json=None, record=None, index=None,
                     **dummy_kwargs):
    """Connect to before_record_index signal to transform record for ES."""
    if index and index.startswith('grants-'):
        # Generate suggest field
        suggestions = [
            json.get('code'),
            json.get('acronym'),
            json.get('title')
        ]
        json['suggest'] = {
            'input': [s for s in suggestions if s],
            'output': json['title'],
            'context': {
                'funder': [json['funder']['doi']]
            },
            'payload': {
                'id': json['internal_id'],
                'legacy_id': (json['code'] if json.get('program') == 'FP7'
                              else json['internal_id']),
                'code': json['code'],
                'title': json['title'],
                'acronym': json.get('acronym'),
                'program': json.get('program'),
            },
        }
    elif index and index.startswith('funders-'):
        # Generate suggest field
        suggestions = json.get('acronyms', []) + [json.get('name')]
        json['suggest'] = {
            'input': [s for s in suggestions if s],
            'output': json['name'],
            'payload': {
                'id': json['doi']
            },
        }