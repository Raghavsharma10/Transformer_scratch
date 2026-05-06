def _dset_info_afni(dset):
    ''' returns raw output from running ``3dinfo`` '''
    info = DsetInfo()
    try:
        raw_info = subprocess.check_output(['3dinfo','-verb',str(dset)],stderr=subprocess.STDOUT)
    except:
        return None
    if raw_info==None:
        return None
    # Subbrick info:
    sub_pattern = r'At sub-brick #(\d+) \'([^\']+)\' datum type is (\w+)(:\s+(.*)\s+to\s+(.*))?\n(.*statcode = (\w+);  statpar = (.*)|)'
    sub_info = re.findall(sub_pattern,raw_info)
    for brick in sub_info:
        brick_info = {
            'index': int(brick[0]),
            'label': brick[1],
            'datum': brick[2]
        }
        if brick[3]!='':
            brick_info.update({
                'min': float(brick[4]),
                'max': float(brick[5])
            })
        if brick[6]!='':
            brick_info.update({
                'stat': brick[7],
                'params': brick[8].split()
            })
        info.subbricks.append(brick_info)
    info.reps = len(info.subbricks)
    # Dimensions:

    orient = re.search('\[-orient ([A-Z]+)\]',raw_info)
    if orient:
        info.orient = orient.group(1)
    for axis in ['RL','AP','IS']:
        m = re.search(r'%s-to-%s extent:\s+([0-9-.]+) \[.\] -to-\s+([0-9-.]+) \[.\] -step-\s+([0-9-.]+) mm \[\s*([0-9]+) voxels\]' % (axis[0],axis[1]),raw_info)
        if m:
            info.spatial_from.append(float(m.group(1)))
            info.spatial_to.append(float(m.group(2)))
            info.voxel_size.append(float(m.group(3)))
            info.voxel_dims.append(float(m.group(4)))
    if len(info.voxel_size)==3:
        info.voxel_volume = reduce(mul,info.voxel_size)

    slice_timing = re.findall('-time:[tz][tz] \d+ \d+ [0-9.]+ (.*?) ',raw_info)
    if len(slice_timing):
        info.slice_timing = slice_timing[0]
    TR = re.findall('Time step = ([0-9.]+)s',raw_info)
    if len(TR):
        info.TR = float(TR[0])

    # Other info..
    details_regex = {
        'identifier': r'Identifier Code:\s+(.*)',
        'filetype': r'Storage Mode:\s+(.*)',
        'space': r'Template Space:\s+(.*)'
    }
    for d in details_regex:
        m = re.search(details_regex[d],raw_info)
        if m:
            setattr(info,d,m.group(1))

    return info