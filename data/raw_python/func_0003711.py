def configure_server():
    '''
    Configure the transfer environment and store
    '''

    home = os.path.expanduser('~')
    if os.path.isfile(os.path.join(home, '.transfer', 'config.yaml')):
        with open(os.path.join(home, '.transfer', 'config.yaml'), 'r') as fp:
            config = yaml.load(fp.read())
    else:
        config = []

    project_name = input('Name your project: ')
    existing_project = None
    for project in config:
        if project_name == project['name']:
            existing_project = project_name
    if existing_project is not None:
        print(colored('Project ' + project_name + ' already exists', 'red'))
        overwrite = str_input('Would you like to overwrite this project? (yes or no) ', ['yes', 'no'])
        if overwrite == 'no':
            return
        else:
            config = [project for project in config if project_name != project['name']]

    api_port = int_input('port for local prediction API (suggested: 5000)', 1024, 49151)
    print('Select image resolution:')
    print('[0] low (224 px)')
    print('[1] mid (448 px)')
    print('[2] high (896 px)')
    img_resolution_index = int_input('choice', 0, 2, show_range = False)
    if img_resolution_index == 0:
        img_size = 1
    elif img_resolution_index == 1:
        img_size = 2
    else:
        img_size = 4
    num_categories = int_input('number of image categories in your model', 0, 10000000)

    weights = False
    while weights == False:
        server_weights = os.path.expanduser(input('Select weights file: '))
        if os.path.isfile(server_weights):
            weights = True
        else:
            print('Cannot find the weight file: ', server_weights)

    project = {'name': project_name,
               'api_port': api_port,
               'img_size': img_size,
               'number_categories': num_categories,
               'server_weights': server_weights}

    config.append(project)
    store_config(config)
    print('')
    print(colored('Project configure saved!', 'cyan'))
    print('')
    print('To start the server:')
    print('')
    print(colored('    transfer --prediction-rest-api --project ' + project_name, 'green'))
    print('or')
    print(colored('    transfer --prediction-rest-api -p ' + project_name, 'green'))