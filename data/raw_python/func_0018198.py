def maybe(value):
    """Wraps an object with a Maybe instance.

      >>> maybe("I'm a value")
      Something("I'm a value")

      >>> maybe(None);
      Nothing

      Testing for value:

        >>> maybe("I'm a value").is_some()
        True
        >>> maybe("I'm a value").is_none()
        False
        >>> maybe(None).is_some()
        False
        >>> maybe(None).is_none()
        True

      Simplifying IF statements:

        >>> maybe("I'm a value").get()
        "I'm a value"

        >>> maybe("I'm a value").or_else(lambda: "No value")
        "I'm a value"

        >>> maybe(None).get()
        Traceback (most recent call last):
        ...
        NothingValueError: No such element

        >>> maybe(None).or_else(lambda: "value")
        'value'

        >>> maybe(None).or_else("value")
        'value'

      Wrap around values from object's attributes:

        class Person(object):
            def __init__(name):
                self.eran = name

        eran = maybe(Person('eran'))

        >>> eran.name
        Something('eran')
        >>> eran.phone_number
        Nothing
        >>> eran.phone_number.or_else('no phone number')
        'no phone number'

        >>> maybe(4) + 8
        Something(12)
        >>> maybe(4) - 2
        Something(2)
        >>> maybe(4) * 2
        Something(8)

      And methods:

        >>> maybe('VALUE').lower().get()
        'value'
        >>> maybe(None).invalid().method().or_else('unknwon')
        'unknwon'

      Enabled easily using NestedDictionaries without having to worry
      if a value is missing.
      For example lets assume we want to load some value from the
      following dictionary:
        nested_dict = maybe({
            'store': {
                'name': 'MyStore',
                    'departments': {
                    'sales': { 'head_count': '10' }
                }
            }
        })

        >>> nested_dict['store']['name'].get()
        'MyStore'
        >>> nested_dict['store']['address']
        Nothing
        >>> nested_dict['store']['address']['street'].or_else('No Address Specified')
        'No Address Specified'
        >>> nested_dict['store']['address']['street'].or_none() is None
        True
        >>> nested_dict['store']['address']['street'].or_empty_list()
        []
        >>> nested_dict['store']['departments']['sales']['head_count'].or_else('0')
        '10'
        >>> nested_dict['store']['departments']['marketing']['head_count'].or_else('0')
        '0'

    """
    if isinstance(value, Maybe):
        return value

    if value is not None:
        return Something(value)

    return Nothing()