def _load_referenced_schema_from_properties(val, a_scheme, a_property):
    """
    :return: updated scheme
    """
    scheme = copy.copy(a_scheme)
    if _value_properties_are_referenced(val):
        ref_schema_uri = val['properties']['$ref']
        sub_schema = load_ref_schema(ref_schema_uri)
        ## dereference the sub schema
        sub_schema_copy_level_0 = copy.deepcopy(sub_schema)

        # nesting level 1
        # @TODO: NEEDS REFACTOR
        for prop_0 in sub_schema_copy_level_0['properties']:
            val_0 = sub_schema_copy_level_0['properties'][prop_0]
            # arrays may contain the nesting
            is_type_array_0 = (val_0['type'] == 'array')
            is_type_object_0 = (val_0['type'] == 'object')
            if ((is_type_array_0 or is_type_object_0)
                and (_value_properties_are_referenced(val_0))):
                # found a nested  thingy
                sub_schema_copy_level_1 = _load_referenced_schema_from_properties(val_0, sub_schema_copy_level_0,
                                                                                  prop_0)
                ###
                # one more loop level
                ###
                for prop_1 in sub_schema_copy_level_1['properties']:
                    val_1 = sub_schema_copy_level_1['properties'][prop_1]
                    is_type_array_1 = (val_1['type'] == 'array')
                    is_type_object_1 = (val_1['type'] == 'object')
                    if ((is_type_array_1 or is_type_object_1) and
                            (_value_properties_are_referenced(val_1))):
                        ### need to figure out a better way for loading
                        # the referenced stuff
                        # found a nested  thingy
                        # sub_schema_copy_level_2 = _load_referenced_schema_from_properties(val_1, sub_schema_copy_level_1, prop_1)
                        raise SalesKingException("too much nesting in the schemes")

            if _value_is_required(val_0):
                # remove required
                sub_schema_copy_level_0['properties'][prop_0]['required'] = False
            # hack to bypass text format valitation to string
            if _value_is_type_text(val_0):
                log.debug("patched text to string")
                sub_schema_copy_level_0['properties'][prop_0]['type'] = u"string"


        # outer scheme
        scheme['properties'][a_property]['properties'] = sub_schema_copy_level_0['properties']

    return scheme