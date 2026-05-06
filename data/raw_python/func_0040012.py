def __compact_notification_addresses(addresses_):
    """
    Input::

        {
            'default': 'http://domain/notice',
            'event1': 'http://domain/notice',
            'event2': 'http://domain/notice',
            'event3': 'http://domain/notice3',
            'event4': 'http://domain/notice3'
        }

    Output::

        {
            'default': 'http://domain/notice',
            'event1,event2': 'http://domain/notice',
            'event3,event4': 'http://domain/notice3'
        }

    """
    result = {}
    addresses = dict(addresses_)
    default = addresses.pop('default', None)

    for key, value in __reverse_group_dict(addresses).items():
        result[','.join(value)] = key

    if default:
        result['default'] = default

    return result