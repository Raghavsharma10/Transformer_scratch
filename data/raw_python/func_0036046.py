def directory():
    """Construct hitman_dir from os name"""
    home = os.path.expanduser('~')
    if platform.system() == 'Linux':
        hitman_dir = os.path.join(home, '.hitman')
    elif platform.system() == 'Darwin':
        hitman_dir = os.path.join(home, 'Library', 'Application Support',
                                  'hitman')
    elif platform.system() == 'Windows':
        hitman_dir = os.path.join(os.environ['appdata'], 'hitman')
    else:
        hitman_dir = os.path.join(home, '.hitman')
    if not os.path.isdir(hitman_dir):
        os.mkdir(hitman_dir)
    return hitman_dir