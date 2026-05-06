def select_project(user_provided_project):
    '''
    Select a project from configuration to run transfer on

    args:
        user_provided_project (str): Project name that should match a project in the config

    returns:
        project (dict): Configuration settings for a user selected project

    '''

    home = os.path.expanduser('~')
    if os.path.isfile(os.path.join(home, '.transfer', 'config.yaml')):
        with open(os.path.join(home, '.transfer', 'config.yaml'), 'r') as fp:
            projects = yaml.load(fp.read())
        if len(projects) == 1:
            project = projects[0]
        else:
            if user_provided_project in [project['name'] for project in projects]:
                for inner_project in projects:
                    if user_provided_project == inner_project['name']:
                        project = inner_project
            else:
                print('Select your project')
                for i, project in enumerate(projects):
                    print('[' + str(i) + ']: ' + project['name'])
                project_index = int_input('project', -1, len(projects), show_range = False)
                project = projects[project_index]
    else:
        print('Transfer is not configured.')
        print('Please run:')
        print('')
        print(colored('    transfer --configure', 'green'))
        return

    print(colored('Project selected: ' + project['name'], 'cyan'))
    return project