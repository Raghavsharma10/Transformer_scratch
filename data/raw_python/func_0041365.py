def load_schema(name):
    """ 
    loads the schema by name
    :param name name of the model
    """

    schema = import_schema_to_json(name)

    #salesking specific swap
    #//set link relation as key name to make it easier to call these
    for item in schema['links']:
        #//set link relation as key name to make it easier to call these
        #            foreach($schema->links as $key => $link)
        #            {
        #                $schema->links[$link->rel] = $link;
        #                unset($schema->links[$key]);
        #            }
        # this here seems not to work as expected
        # something is wrong
        href_value = item['href']
        rel_value = item['rel']
        schema[rel_value] = href_value
        del item

    ## sk use nesting of schema
    ## dynamically loading
    for prop in schema['properties']:
        value = schema['properties'][prop]
        # arrays may contain the nesting
        is_type_array = (value['type'] == 'array')
        is_type_object = (value['type'] == 'object')
        if ((is_type_array or is_type_object)
            and (_value_properties_are_referenced(value))):
            schema = _load_referenced_schema_from_properties(value, schema, prop)

        if is_type_array and _value_is_default_any(value) and _value_has_items_key(value):
            schema = _load_referenced_schemes_from_list(value['items'], value, schema, prop)

        if _value_is_required(value):
            # remove required
            schema['properties'][prop]['required'] = False
        
        # hack to bypass text format valitation to string
        if _value_is_type_text(value):
            log.debug("patched text to string")
            schema['properties'][prop]['type'] = u"string"
        
        #ignore the readonly properties auto validation
        #if 'readonly' in value.keys() and value['readonly'] == True:
        #    log.debug("patched required validation to none required")
        #    schema['properties'][property]['readonly'] = False

    # sk works on title and not name
    schema['name'] = schema['title']
    ## go one level deeper as we now have some replacements

    # put it to storage when done
    # if not JsonSchemaStore.is_stored(name) and (schema is not None):
    #    JsonSchemaStore.copy_to_store(name, schema)
    return schema