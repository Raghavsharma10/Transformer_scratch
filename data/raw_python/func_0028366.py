def is_legacy_server():
    '''Determine execution mode.

    Legacy mode: <= v4.20181215
    '''
    with Session() as session:
        ret = session.Kernel.hello()
    bai_version = ret['version']
    legacy = True if bai_version <= 'v4.20181215' else False
    return legacy