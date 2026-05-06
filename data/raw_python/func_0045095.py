def entityTriples(rdfGraph, anEntity, excludeProps=False, excludeBNodes=False,
                  orderProps=[RDF, RDFS, OWL.OWLNS, DC.DCNS]):
    """
    Returns the pred-obj for any given resource, excluding selected ones..

    Sorting: by default results are sorted alphabetically and according to namespaces: [RDF, RDFS, OWL.OWLNS, DC.DCNS]
    """
    temp = []
    if not excludeProps:
        excludeProps = []

    # extract predicate/object
    for x, y, z in rdfGraph.triples((anEntity, None, None)):
        if excludeBNodes and isBlankNode(z):
            continue
        if y not in excludeProps:
            temp += [(y, z)]

    # sorting
    if type(orderProps) == type([]):
        orderedUris = sortByNamespacePrefix([y for y, z in temp], orderProps)  # order props only
        orderedUris = [(n + 1, x) for n, x in
                       enumerate(orderedUris)]  # create form: [(1, 'something'),(2,'bobby'),(3,'suzy'),(4,'crab')]
        rank = dict((key, rank) for (rank, key) in orderedUris)  # create dict to pass to sorted procedure
        temp = sorted(temp, key=lambda tup: rank.get(tup[0]))
    elif orderProps:  # default to alpha sorting unless False
        temp = sorted(temp, key=lambda tup: tup[0])

    # if niceURI:
    #	temp = [(uri2niceString(ontology, y), z) for y,z in temp]

    return temp