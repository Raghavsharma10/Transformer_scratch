def info(filename):
    '''returns a DicomInfo object containing the header information in ``filename``'''
    try:
        out = subprocess.check_output([_dicom_hdr,'-sexinfo',filename])
    except subprocess.CalledProcessError:
        return None
    slice_timing_out = subprocess.check_output([_dicom_hdr,'-slice_times',filename])
    slice_timing = [float(x) for x in slice_timing_out.strip().split()[5:]]
    frames = []
    for frame in re.findall(r'^(\w{4}) (\w{4})\s+(\d+) \[(\d+)\s+\] \/\/(.*?)\/\/(.*?)$',out,re.M):
        new_frame = {}
        new_frame['addr'] = (int(frame[0],16),int(frame[1],16))
        new_frame['size'] = int(frame[2])
        new_frame['offset'] = int(frame[3])
        new_frame['label'] = frame[4].strip()
        new_frame['value'] = frame[5].strip()
        frames.append(new_frame)
    sex_info = {}
    for i in re.findall(r'^(.*?)\s+= (.*)$',out,re.M):
        sex_info[i[0]] = i[1]

    return DicomInfo(frames,sex_info,slice_timing)