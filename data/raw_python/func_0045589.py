def _gen_figure(nxplot=1, nyplot=1, figargs=None, projection=None,
                sharex='none', joinx=False, sharey='none', joiny=False,
                x=None, nxlabel=None, xlabels=None, nxdecimal=None, xmin=None, xmax=None,
                y=None, nylabel=None, ylabels=None, nydecimal=None, ymin=None, ymax=None,
                z=None, nzlabel=None, zlabels=None, nzdecimal=None, zmin=None, zmax=None,
                r=None, nrlabel=None, rlabels=None, nrdecimal=None, rmin=None, rmax=None,
                t=None, ntlabel=None, tlabels=None, fontsize=20):
    """
    Returns a figure object with as much customization as provided.
    """
    figargs = {} if figargs is None else figargs
    if projection is not None:
        fig, axs = _gen_projected(nxplot, nyplot, projection, figargs)
    else:
        fig, axs = _gen_shared(nxplot, nyplot, sharex, sharey, figargs)
    adj = {}
    if joinx: adj.update({'hspace': 0})
    if joiny: adj.update({'wspace': 0})
    fig.subplots_adjust(**adj)
    data = {}
    if projection is None:
        data = {'x': x, 'y': y}
    elif projection == '3d':
        data = {'x': x, 'y': y, 'z': z}
    elif projection == 'polar':
        data = {'r': r, 't': t}
    methods = {}
    for ax in axs:
        if 'x' in data:
            methods['x'] = (ax.set_xlim, ax.set_xticks, ax.set_xticklabels,
                            nxlabel, xlabels, nxdecimal, xmin, xmax)
        if 'y' in data:
            methods['y'] = (ax.set_ylim, ax.set_yticks, ax.set_yticklabels,
                            nylabel, ylabels, nydecimal, ymin, ymax)
        if 'z' in data:
            methods['z'] = (ax.set_zlim, ax.set_zticks, ax.set_zticklabels,
                            nzlabel, zlabels, nzdecimal, zmin, zmax)
        if 'r' in data:
            methods['r'] = (ax.set_rlim, ax.set_rticks, ax.set_rgrids,
                            nrlabel, rlabels, nrdecimal, rmin, rmax)
        if 't' in data:
            methods['t'] = (ax.set_thetagrids, ntlabel, tlabels)
        for dim, arr in data.items():
            if dim == 't':
                grids, nlabel, labls = methods[dim]
                if ntlabel is not None:
                    theta = np.arange(0, 2 * np.pi, 2 * np.pi / ntlabel)
                    if labls is not None:
                        grids(np.degrees(theta), labls, fontsize=fontsize)
                    else:
                        grids(np.degrees(theta), fontsize=fontsize)
            else:
                lim, ticks, labels, nlabel, labls, decs, mins, maxs = methods[dim]
                if arr is not None:
                    amin = mins if mins is not None else arr.min()
                    amax = maxs if maxs is not None else arr.max()
                    lim((amin, amax))
                elif mins is not None and maxs is not None:
                    if nlabel is not None:
                        ticks(np.linspace(amin, amax, nlabel))
                        if decs is not None:
                            sub = "{{:.{}f}}".format(decs).format
                            labels([sub(i) for i in np.linspace(amin, amax, nlabel)])
                if labls is not None:
                    labels(labls)
                ax.tick_params(axis=dim, labelsize=fontsize)
    return fig