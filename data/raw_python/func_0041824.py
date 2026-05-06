def load_config(options):
    ''' Load options, platform, colors, and icons. '''
    global opts, pform
    opts = options
    pform = options.pform
    global_ns = globals()

    # get colors
    if pform.hicolor:
        global_ns['dim_templ'] = ansi.dim8t
        global_ns['swap_clr_templ'] = ansi.csi8_blk % ansi.blu8
    else:
        global_ns['dim_templ'] = ansi.dim4t
        global_ns['swap_clr_templ'] = ansi.fbblue

    # load icons into module namespace
    for varname in dir(pform):
        if varname.startswith('_') and varname.endswith('ico'):
            global_ns[varname] = getattr(pform, varname)