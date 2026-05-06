def open_vocab_values(instance):
    """Ensure that the values of all properties which use open vocabularies are
    in lowercase and use hyphens instead of spaces or underscores as word
    separators.
    """
    if instance['type'] not in enums.VOCAB_PROPERTIES:
        return

    properties = enums.VOCAB_PROPERTIES[instance['type']]
    for prop in properties:
        if prop in instance:

            if type(instance[prop]) is list:
                values = instance[prop]
            else:
                values = [instance[prop]]

            for v in values:
                if not v.islower() or '_' in v or ' ' in v:
                    yield JSONError("Open vocabulary value '%s' should be all"
                                    " lowercase and use hyphens instead of"
                                    " spaces or underscores as word"
                                    " separators." % v, instance['id'],
                                    'open-vocab-format')