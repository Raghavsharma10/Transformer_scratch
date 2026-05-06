def _get_project(msg, key='project'):
    ''' Return the project as `foo` or `user/foo` if the project is a
    fork.
    '''
    project = msg[key]['name']
    ns = msg[key].get('namespace')
    if ns:
        project = '/'.join([ns, project])
    if msg[key]['parent']:
        user = msg[key]['user']['name']
        project = '/'.join(['fork', user, project])
    return project