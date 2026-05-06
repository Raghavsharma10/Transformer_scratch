def get_all_files_in_range(dirname, starttime, endtime, pad=64):
    """Returns all files in dirname and all its subdirectories whose
    names indicate that they contain segments in the range starttime
    to endtime"""
    
    ret = []

    # Maybe the user just wants one file...
    if os.path.isfile(dirname):
        if re.match('.*-[0-9]*-[0-9]*\.xml$', dirname):
            return [dirname]
        else:
            return ret

    first_four_start = starttime / 100000
    first_four_end   = endtime   / 100000

    for filename in os.listdir(dirname):
        if re.match('.*-[0-9]{5}$', filename):
            dirtime = int(filename[-5:])
            if dirtime >= first_four_start and dirtime <= first_four_end:
                ret += get_all_files_in_range(os.path.join(dirname,filename), starttime, endtime, pad=pad)
        elif re.match('.*-[0-9]{4}$', filename):
            dirtime = int(filename[-4:])
            if dirtime >= first_four_start and dirtime <= first_four_end:
                ret += get_all_files_in_range(os.path.join(dirname,filename), starttime, endtime, pad=pad)
        elif re.match('.*-[0-9]*-[0-9]*\.xml$', filename):
            file_time = int(filename.split('-')[-2])
            if file_time >= (starttime-pad) and file_time <= (endtime+pad):
                ret.append(os.path.join(dirname,filename))
        else:
            # Keep recursing, we may be looking at directories of
            # ifos, each of which has directories with times
            ret += get_all_files_in_range(os.path.join(dirname,filename), starttime, endtime, pad=pad)

    return ret