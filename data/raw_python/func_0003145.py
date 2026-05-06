def draw_graph( g, fmt='svg', prg='dot', options={} ):
    """
    Draw an RDF graph as an image
    """
    # Convert RDF to Graphviz
    buf = StringIO()
    rdf2dot( g, buf, options )

    gv_options = options.get('graphviz',[])
    if fmt == 'png':
        gv_options += [ '-Gdpi=220', '-Gsize=25,10!' ]
        metadata = { "width": 5500, "height": 2200, "unconfined" : True }

    #import codecs
    #with codecs.open('/tmp/sparqlkernel-img.dot','w',encoding='utf-8') as f:
    #    f.write( buf.getvalue() )
    
    # Now use Graphviz to generate the graph
    image = run_dot( buf.getvalue(), fmt=fmt, options=gv_options, prg=prg )

    #with open('/tmp/sparqlkernel-img.'+fmt,'w') as f:
    #    f.write( image )

    # Return it
    if fmt == 'png':
        return { 'image/png' : base64.b64encode(image).decode('ascii') }, \
               { "image/png" : metadata }
    elif fmt == 'svg':
        return { 'image/svg+xml' : image.decode('utf-8').replace('<svg','<svg class="unconfined"',1) }, \
               { "unconfined" : True }