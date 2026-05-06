def get_edge_mark(ttree):
    """ makes a simple Graph Mark object"""
    
    ## tree style
    if ttree._kwargs["tree_style"] in ["c", "cladogram"]:
        a=ttree.edges
        vcoordinates=ttree.verts
    else:
        a=ttree._lines               
        vcoordinates=ttree._coords    
   
    ## fixed args
    along='x'
    vmarker='o'
    vcolor=None
    vlshow=False            
    vsize=0.         
    estyle=ttree._kwargs["edge_style"]

    ## get axes
    layout = toyplot.layout.graph(a, vcoordinates=vcoordinates)
    along = toyplot.require.value_in(along, ["x", "y"])
    if along == "x":
        coordinate_axes = ["x", "y"]
    elif along == "y":
        coordinate_axes = ["y", "x"]
        
    ## broadcast args along axes
    vlabel = layout.vids
    vmarker = toyplot.broadcast.pyobject(vmarker, layout.vcount)
    vsize = toyplot.broadcast.scalar(vsize, layout.vcount)
    estyle = toyplot.style.require(estyle, allowed=toyplot.style.allowed.line)

    ## fixed args
    vcolor = toyplot.color.broadcast(colors=None, shape=layout.vcount, default=toyplot.color.black)
    vopacity = toyplot.broadcast.scalar(1.0, layout.vcount)
    vtitle = toyplot.broadcast.pyobject(None, layout.vcount)
    vstyle = None
    vlstyle = None
    
    ## this could be modified in the future to allow diff color edges
    ecolor = toyplot.color.broadcast(colors=None, shape=layout.ecount, default=toyplot.color.black)
    ewidth = toyplot.broadcast.scalar(1.0, layout.ecount)
    eopacity = toyplot.broadcast.scalar(1.0, layout.ecount)
    hmarker = toyplot.broadcast.pyobject(None, layout.ecount)
    mmarker = toyplot.broadcast.pyobject(None, layout.ecount)
    mposition = toyplot.broadcast.scalar(0.5, layout.ecount)
    tmarker = toyplot.broadcast.pyobject(None, layout.ecount)
    
    ## tables are required if I don't want to edit the class
    vtable = toyplot.data.Table()
    vtable["id"] = layout.vids
    for axis, coordinates in zip(coordinate_axes, layout.vcoordinates.T):
        vtable[axis] = coordinates
        #_mark_exportable(vtable, axis)
    vtable["label"] = vlabel
    vtable["marker"] = vmarker
    vtable["size"] = vsize
    vtable["color"] = vcolor
    vtable["opacity"] = vopacity
    vtable["title"] = vtitle

    etable = toyplot.data.Table()
    etable["source"] = layout.edges.T[0]
    #_mark_exportable(etable, "source")
    etable["target"] = layout.edges.T[1]
    #_mark_exportable(etable, "target")
    etable["shape"] = layout.eshapes
    etable["color"] = ecolor
    etable["width"] = ewidth
    etable["opacity"] = eopacity
    etable["hmarker"] = hmarker
    etable["mmarker"] = mmarker
    etable["mposition"] = mposition
    etable["tmarker"] = tmarker
    
    edge_mark = toyplot.mark.Graph(
        coordinate_axes=['x', 'y'],
        ecolor=["color"],
        ecoordinates=layout.ecoordinates,
        efilename=None,
        eopacity=["opacity"],
        eshape=["shape"],
        esource=["source"],
        estyle=estyle,
        etable=etable,
        etarget=["target"],
        ewidth=["width"],
        hmarker=["hmarker"],
        mmarker=["mmarker"],
        mposition=["mposition"],
        tmarker=["tmarker"],
        vcolor=["color"],
        vcoordinates=['x', 'y'],
        vfilename=None,
        vid=["id"],
        vlabel=["label"],
        vlshow=False,
        vlstyle=None,
        vmarker=["marker"],
        vopacity=["opacity"],
        vsize=["size"],
        vstyle=None,
        vtable=vtable,
        vtitle=["title"],
        )
    return edge_mark