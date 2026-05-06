def get_info(line, bit_thresh):
    """
    get info from either ssu-cmsearch or cmsearch output
    """
    if len(line) >= 18: # output is from cmsearch
        id, model, bit, inc = line[0].split()[0], line[2], float(line[14]), line[16]
        sstart, send, strand = int(line[7]), int(line[8]), line[9]
        mstart, mend = int(line[5]), int(line[6])
    elif len(line) == 9: # output is from ssu-cmsearch
        if bit_thresh == 0:
            print('# ssu-cmsearch does not include a model-specific inclusion threshold, ', file=sys.stderr)
            print('# please specify a bit score threshold', file=sys.stderr)
            exit()
        id, model, bit = line[1].split()[0], line[0], float(line[6])
        inc = '!' # this is not a feature of ssu-cmsearch
        sstart, send = int(line[2]), int(line[3])
        mstart, mend = int(4), int(5)
        if send >= sstart:
            strand = '+'
        else:
            strand = '-'
    else:
        print('# unsupported hmm format:', file=sys.stderr)
        print('# provide tabular output from ssu-cmsearch and cmsearch supported', file=sys.stderr)
        exit()
    coords = [sstart, send]
    sstart, send = min(coords), max(coords)
    mcoords = [mstart, mend]
    mstart, mend = min(mcoords), max(mcoords)
    return id, model, bit, sstart, send, mstart, mend, strand, inc