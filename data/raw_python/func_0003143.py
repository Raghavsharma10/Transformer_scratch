def label(x, gr, preferred_languages=None):
    """
      @param x : graph entity
      @param gr (Graph): RDF graph
      @param preferred_languages (iterable)

    Return the best available label in the graph for the passed entity.
    If a set of preferred languages is given, try them in order. If none is
    found, an arbitrary language will be chosen
    """
    # Find all labels & their language
    labels = { l.language : l
               for labelProp in LABEL_PROPERTIES
               for l in gr.objects(x,labelProp) }
    if labels:
        #return repr(preferred_languages) + repr(labels)
        #return u'|'.join(preferred_languages) +  u' -> ' + u'/'.join( u'{}:{}'.format(*i) for i in labels.items() )
        if preferred_languages is not None:
            for l in preferred_languages:
                if l in labels:
                    return labels[l]
        return labels.itervalues().next()

    # No labels available. Try to generate a QNAME, or else, the string itself
    try:
        return gr.namespace_manager.compute_qname(x)[2].replace('_',' ')
    except:
        # Attempt to extract the trailing part of an URI
        m = re.search( '([^/]+)$', x )
        return m.group(1).replace('_',' ') if m else x