def triplify_object(binding):
    """ Create bi-directional bindings for object relationships. """
    triples = []
    if binding.uri:
        triples.append((binding.subject, RDF.type, binding.uri))

    if binding.parent is not None:
        parent = binding.parent.subject
        if binding.parent.is_array:
            parent = binding.parent.parent.subject
        triples.append((parent, binding.predicate, binding.subject))
        if binding.reverse is not None:
            triples.append((binding.subject, binding.reverse, parent))

    for prop in binding.properties:
        _, prop_triples = triplify(prop)
        triples.extend(prop_triples)

    return binding.subject, triples