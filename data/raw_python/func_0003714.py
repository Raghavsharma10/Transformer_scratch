def update_config(updated_project):
    '''
    Update project in configuration

    args:
        updated_project (dict): Updated project configuration values

    '''

    home = os.path.expanduser('~')
    if os.path.isfile(os.path.join(home, '.transfer', 'config.yaml')):
        with open(os.path.join(home, '.transfer', 'config.yaml'), 'r') as fp:
            projects = yaml.load(fp.read())
        replace_index = -1
        for i, project in enumerate(projects):
            if project['name'] == updated_project['name']:
                replace_index = i

        if replace_index > -1:
            projects[replace_index] = updated_project
            store_config(projects)
        else:
            print('Not saving configuration')
            print(colored('Project: ' + updated_project['name'] + ' was not found in configured projects!', 'red'))

    else:
        print('Transfer is not configured.')
        print('Please run:')
        print('')
        print(colored('    transfer --configure', 'cyan'))
        return