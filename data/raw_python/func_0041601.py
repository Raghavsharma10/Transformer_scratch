def check_optical(disk):
    ''' Try to determine if a device is optical technology.
        Needs improvement.
    '''
    dev = disk.dev
    if dev.startswith('sr') or ('cd' in dev):
        return True
    elif disk.fmt in optical_fs:
        return True
    else:
        return None