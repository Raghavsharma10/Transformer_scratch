def split_styles(mark):
    """ get shared styles """
    
    markers = [mark._table[key] for key in mark._marker][0]
    nstyles = []
    for m in markers:
        ## fill and stroke are already rgb() since already in markers
        msty = toyplot.style.combine({
            "fill": m.mstyle['fill'],
            "stroke": m.mstyle['stroke'],
            "opacity": m.mstyle["fill-opacity"],
        }, m.mstyle)
        msty = _color_fixup(msty)
        nstyles.append(msty)
    
    ## uses 'marker.size' so we need to loop over it
    lstyles = []
    for m in markers:
        lsty = toyplot.style.combine({
        "font-family": "Helvetica",
        "-toyplot-vertical-align": "middle",
        "fill": toyplot.color.black,
        "font-size": "%rpx" % (m.size * 0.75),
        "stroke": "none",
        "text-anchor": "middle",
        }, m.lstyle)
        ## update fonts
        fonts = toyplot.font.ReportlabLibrary()
        layout = toyplot.text.layout(m.label, lsty, fonts)
        lsty = _color_fixup(layout.style)
        lstyles.append(lsty)
    
    nallkeys = set(itertools.chain(*[i.keys() for i in nstyles]))
    lallkeys = set(itertools.chain(*[i.keys() for i in lstyles]))
    nuniquekeys = []
    nsharedkeys = []
    for key in nallkeys:
        vals = [nstyles[i].get(key) for i in range(len(nstyles))]
        if len(set(vals)) > 1:
            nuniquekeys.append(key)
        else:
            nsharedkeys.append(key)
    luniquekeys = []
    lsharedkeys = []
    for key in lallkeys:
        vals = [lstyles[i].get(key) for i in range(len(lstyles))]
        if len(set(vals)) > 1:
            luniquekeys.append(key)
        else:
            lsharedkeys.append(key)

    ## keys shared between mark and text markers
    repeated = set(lsharedkeys).intersection(set(nsharedkeys))
    for repeat in repeated:
        ## if same then keep only one copy of it
        lidx = lsharedkeys.index(repeat)
        nidx = nsharedkeys.index(repeat)
        if lsharedkeys[lidx] == nsharedkeys[nidx]:
            lsharedkeys.remove(repeat)
        else:
            lsharedkeys.remove(repeat)
            luniquekeys.append(repeat)
            nsharedkeys.remove(repeat)
            nuniquekeys.append(repeat)
            
    ## check node values
    natt = ["%s:%s" % (key, nstyles[0][key]) for key in sorted(nsharedkeys)]
    latt = ["%s:%s" % (key, lstyles[0][key]) for key in sorted(lsharedkeys)]
    shared_styles = ";".join(natt+latt)
    unique_styles = {
        "node": [{k:v for k,v in nstyles[idx].items() if k in nuniquekeys} for idx in range(len(markers))],
        "text": [{k:v for k,v in lstyles[idx].items() if k in luniquekeys} for idx in range(len(markers))]
    }
    
    return shared_styles, unique_styles