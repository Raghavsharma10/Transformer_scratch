def print_diskinfo(diskinfo, widelayout, incolor):
    ''' Disk information output function. '''
    sep = ' '
    if opts.relative:
        import math
        base = max([ disk.ocap for disk in diskinfo ])

    for disk in diskinfo:
        if disk.ismntd:     ico = _diskico
        else:               ico = _unmnico
        if disk.isrem:      ico = _remvico
        if disk.isopt:      ico = _discico
        if disk.isnet:      ico = _netwico
        if disk.isram:      ico = _ramico
        if disk.isimg:      ico = _imgico
        if disk.mntp == '/boot/efi':
                            ico = _gearico

        if opts.relative and disk.ocap and disk.ocap != base:
            # increase log size reduction by raising to 4th power:
            gwidth = int((math.log(disk.ocap, base)**4) * opts.width)
        else:
            gwidth = opts.width

        # check color settings, ffg: free foreground, ufg: used forground
        if disk.rw:
            ffg = ufg = None        # auto colors
        else:
            # dim or dark grey
            ffg = ufg = (ansi.dim8 if opts.hicolor else ansi.dim4)

        cap = disk.cap
        if cap and disk.rw:
            lblcolor = ansi.get_label_tmpl(disk.pcnt, opts.width, opts.hicolor)
        else:
            lblcolor = None

        # print stats
        data = (
            (_usedico, disk.pcnt,     ufg,  None,  pform.boldbar),  # Used
            (_freeico, 100-disk.pcnt, ffg,  None,  False),          # free
        )
        mntp = fmtstr(disk.mntp, align='<', trunc='left',
                      width=(opts.colwidth * 2) + 2)
        mntp = mntp.rstrip()  # prevent wrap
        if disk.label is None:
            label = fmtstr(_emptico, dim_templ, align='<')
        else:
            label = fmtstr(disk.label, align='<')

        if widelayout:
            out(
                fmtstr(ico + sep + disk.dev, align='<') + label
            )
            if cap:
                out(fmtval(cap))
                if disk.rw:
                    out(
                        fmtval(disk.used, lblcolor) +
                        fmtval(disk.free, lblcolor)
                    )
                else:
                    out(
                        fmtstr() +
                        fmtstr(_emptico, dim_templ)
                    )
            else:
                out(fmtstr(_emptico, dim_templ))

            if cap:
                if disk.rw:  # factoring this caused colored brackets
                    ansi.rainbar(data, gwidth, incolor,
                                 hicolor=opts.hicolor,
                                 cbrackets=_brckico)
                else:
                    ansi.bargraph(data, gwidth, incolor, cbrackets=_brckico)

                if opts.relative and opts.width != gwidth:
                    out(sep * (opts.width - gwidth))
                out(sep + mntp)
            print()
        else:
            out(
                fmtstr(ico + sep + disk.dev, align="<") + label
            )
            if cap:
                out(
                    fmtval(cap) +
                    fmtval(disk.used, lblcolor) +
                    fmtval(disk.free, lblcolor)
                )
            else:
                out(fmtstr(_emptico, dim_templ) + fmtstr() + fmtstr())
            print(sep, mntp)

            if cap:
                out(fmtstr())
                if disk.rw:
                    ansi.rainbar(data, gwidth, incolor, hicolor=opts.hicolor,
                                 cbrackets=_brckico)
                else:
                    ansi.bargraph(data, gwidth, incolor, cbrackets=_brckico)
            print()
            print()
    print()