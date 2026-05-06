def Setup():
    '''
    Called when the code is installed. Sets up directories and downloads
    the K2 catalog.

    '''

    if not os.path.exists(os.path.join(EVEREST_DAT, 'k2', 'cbv')):
        os.makedirs(os.path.join(EVEREST_DAT, 'k2', 'cbv'))
    GetK2Stars(clobber=False)