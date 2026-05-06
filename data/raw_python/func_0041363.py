def _load_referenced_schemes_from_list(the_list, val, a_scheme, a_property):
    """
    takes the referenced files and loads them
    returns the updated schema
    :param the_list:
    :param val:
    :param a_scheme:
    :param a_property:
    :return:
    """
    scheme = copy.copy(a_scheme)
    new_list = []
    if isinstance(the_list, list):
        for an_item in the_list:
            if ((not isinstance(an_item, basestring)) and
                    (u'$ref' in an_item.keys())):
                sub_scheme_name = generate_schema_name_from_uri(an_item['$ref'])
                content = load_schema(sub_scheme_name)
                new_list.append(content)
    else:
        # somewhere the array is not an array - payment_reminder
        sub_scheme_name = generate_schema_name_from_uri(the_list['$ref'])
        new_list = load_schema(sub_scheme_name)
    scheme['properties'][a_property]['items'] = new_list
    return scheme