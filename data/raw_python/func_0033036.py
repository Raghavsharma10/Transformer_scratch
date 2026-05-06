def triplify(binding):
    """ Recursively generate RDF statement triples from the data and
    schema supplied to the application. """
    triples = []
    if binding.data is None:
        return None, triples

    if binding.is_object:
        return triplify_object(binding)
    elif binding.is_array:
        for item in binding.items:
            _, item_triples = triplify(item)
            triples.extend(item_triples)
        return None, triples
    else:
        subject = binding.parent.subject
        triples.append((subject, binding.predicate, binding.object))
        if binding.reverse is not None:
            triples.append((binding.object, binding.reverse, subject))
        return subject, triples