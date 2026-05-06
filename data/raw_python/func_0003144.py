def rdf2dot( g, stream, opts={} ):
    """
    Convert the RDF graph to DOT
    Write the dot output to the stream
    """

    accept_lang = set( opts.get('lang',[]) )
    do_literal = opts.get('literal')
    nodes = {}
    links = []

    def node_id(x):
        if x not in nodes:
            nodes[x] = "node%d" % len(nodes)
        return nodes[x]

    def qname(x, g):
        try:
            q = g.compute_qname(x)
            return q[0] + ":" + q[2]
        except:
            return x

    def accept( node ):
        if isinstance( node, (rdflib.URIRef,rdflib.BNode) ):
            return True
        if not do_literal:
            return False
        return (not accept_lang) or (node.language in accept_lang)


    stream.write( u'digraph { \n node [ fontname="DejaVu Sans,Tahoma,Geneva,sans-serif" ] ; \n' )

    # Write all edges. In the process make a list of all nodes
    for s, p, o in g:
        # skip triples for labels
        if p == rdflib.RDFS.label:
            continue

        # Create a link if both objects are graph nodes
        # (or, if literals are also included, if their languages match)
        if not (accept(s) and accept(o)):
            continue

        # add the nodes to the list
        sn = node_id(s)
        on = node_id(o)

        # add the link
        q = qname(p,g)
        if isinstance(p, rdflib.URIRef):
            opstr = u'\t%s -> %s [ arrowhead="open", color="#9FC9E560", fontsize=9, fontcolor="#204080", label="%s", href="%s", target="_other" ] ;\n' % (sn,on,q,p)
        else:
            opstr = u'\t%s -> %s [ arrowhead="open", color="#9FC9E560", fontsize=9, fontcolor="#204080", label="%s" ] ;\n'%(sn,on,q)
        stream.write( opstr )

    # Write all nodes
    for u, n in nodes.items():
        lbl = escape( label(u,g,accept_lang), True )
        if isinstance(u, rdflib.URIRef):
            opstr = u'%s [ shape=none, fontsize=10, fontcolor=%s, label="%s", href="%s", target=_other ] \n' % (n, 'blue', lbl, u )
        else:
            opstr = u'%s [ shape=none, fontsize=10, fontcolor=%s, label="%s" ] \n' % (n, 'black', lbl )
        stream.write( u"# %s %s\n" % (u, n) )
        stream.write( opstr )

    stream.write(u'}\n')