def prange(*x):
    '''
    Progress bar range with `tqdm`

    '''

    try:
        root = logging.getLogger()
        if len(root.handlers):
            for h in root.handlers:
                if (type(h) is logging.StreamHandler) and \
                        (h.level != logging.CRITICAL):
                    from tqdm import tqdm
                    return tqdm(range(*x))
            return range(*x)
        else:
            from tqdm import tqdm
            return tqdm(range(*x))
    except ImportError:
        return range(*x)