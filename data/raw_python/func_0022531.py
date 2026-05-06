def isgood(name):
    """ Whether name should be installed """
    if not isbad(name):
        if name.endswith('.py') or name.endswith('.json') or name.endswith('.tar'):
            return True
    return False