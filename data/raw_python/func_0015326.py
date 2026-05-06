def match(license):
    '''Returns True if given license field is correct
    Taken from rpmlint.

    It's named match() to mimic a compiled regexp.'''
    if license not in VALID_LICENSES:
        for l1 in _split_license(license):
            if l1 in VALID_LICENSES:
                continue
            for l2 in _split_license(l1):
                if l2 not in VALID_LICENSES:
                    return False
                    valid_license = False
    return True