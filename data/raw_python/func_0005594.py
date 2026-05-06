def find_root(filename, target='bids'):
    """Find base directory (root) for a filename.

    Parameters
    ----------
    filename : instance of Path
        search the root for this file
    target: str
        'bids' (the directory containing 'participants.tsv'), 'subject' (the
        directory starting with 'sub-'), 'session' (the directory starting with
        'ses-')

    Returns
    -------
    Path
        path of the target directory
    """
    lg.debug(f'Searching root in {filename}')
    if target == 'bids' and (filename / 'dataset_description.json').exists():
        return filename
    elif filename.is_dir():
        pattern = target[:3] + '-'
        if filename.stem.startswith(pattern):
            return filename

    return find_root(filename.parent, target)