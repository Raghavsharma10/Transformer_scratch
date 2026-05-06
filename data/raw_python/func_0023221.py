def _list_fonts():
    """List system fonts"""
    stdout_, stderr = run_subprocess(['fc-list', ':scalable=true', 'family'])
    vals = [v.split(',')[0] for v in stdout_.strip().splitlines(False)]
    return vals