def DVSFile(ID, season, cadence='lc'):
    '''
    Returns the name of the DVS PDF for a given target.

    :param ID: The target ID
    :param int season: The target season number
    :param str cadence: The cadence type. Default `lc`

    '''

    if cadence == 'sc':
        strcadence = '_sc'
    else:
        strcadence = ''
    return 'hlsp_everest_k2_llc_%d-c%02d_kepler_v%s_dvs%s.pdf' \
           % (ID, season, EVEREST_MAJOR_MINOR, strcadence)