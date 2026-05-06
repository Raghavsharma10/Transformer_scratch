def factorise_strings (string_list, boundary_char=None):
    """Given a list of strings, finds the longest string that is common
    to the *beginning* of all strings in the list and
    returns a new list whose elements lack this common beginning.
    
    boundary_char defines a boundary that must be preserved, so that the
    common string removed must end with this char.
    
    >>> cmn='something/to/begin with?'
    >>> blah=[cmn+'yes',cmn+'no',cmn+'?maybe']
    >>> (blee, bleecmn) = factorise_strings(blah)
    >>> blee
    ['yes', 'no', '?maybe']
    >>> bleecmn == cmn
    True
    
    >>> blah = ['de.uos.nbp.senhance', 'de.uos.nbp.heartFelt']
    >>> (blee, bleecmn) = factorise_strings(blah, '.')
    >>> blee
    ['senhance', 'heartFelt']
    >>> bleecmn
    'de.uos.nbp.'
    
    >>> blah = ['/some/deep/dir/subdir', '/some/deep/other/dir', '/some/deep/other/dir2']
    >>> (blee, bleecmn) = factorise_strings(blah, '/')
    >>> blee
    ['dir/subdir', 'other/dir', 'other/dir2']
    >>> bleecmn
    '/some/deep/'
    
    >>> blah = ['/net/store/nbp/heartFelt/data/ecg/emotive_interoception/p20/2012-01-27T09.01.14-ecg.csv', '/net/store/nbp/heartFelt/data/ecg/emotive_interoception/p21/2012-01-27T11.03.08-ecg.csv', '/net/store/nbp/heartFelt/data/ecg/emotive_interoception/p23/2012-01-31T12.02.55-ecg.csv']
    >>> (blee, bleecmn) = factorise_strings(blah, '/')
    >>> bleecmn
    '/net/store/nbp/heartFelt/data/ecg/emotive_interoception/'
    
    rmuil 2012/02/01
    """
    
    cmn = find_common_beginning(string_list, boundary_char)
    
    new_list = [el[len(cmn):] for el in string_list]

    return (new_list, cmn)