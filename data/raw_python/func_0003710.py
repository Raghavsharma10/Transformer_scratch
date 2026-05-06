def configure():
    '''
    Configure the transfer environment and store
    '''
    completer = Completer()
    readline.set_completer_delims('\t')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(completer.path_completer)

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

    image_path = os.path.expanduser(input('Select parent directory for your images: '))
    path_unset = True
    while path_unset:
        project_path = os.path.expanduser(input('Select destination for your project: '))
        if (project_path.find(image_path) == 0):
            print('Project destination should not be same or within image directory!')
        else:
            path_unset = False

    print('Select architecture:')
    print('[0] resnet50')
    print('[1] xception')
    print('[2] inception_v3')
    architecture = int_input('choice', 0, 2, show_range = False)
    if architecture == 0:
        arch = 'resnet50'
        img_dim = 224
        conv_dim = 7
        final_cutoff = 80
    elif architecture == 1:
        arch = 'xception'
        img_dim = 299
        conv_dim = 10
        final_cutoff = 80
    else:
        arch = 'inception_v3'
        img_dim = 299
        conv_dim = 8
        final_cutoff = 80
    api_port = int_input('port for local prediction API (suggested: 5000)', 1024, 49151)
    kfold = int_input('number of folds to use (suggested: 5)', 3, 10)
    kfold_every = bool_input('Fit a model for every fold? (if false, just fit one)')
    print('Warning: if working on a remote computer, you may not be able to plot!')
    plot_cm = bool_input('Plot a confusion matrix after training?')
    batch_size = int_input('batch size (suggested: 8)', 1, 64)
    learning_rate = float_input('learning rate (suggested: 0.001)', 0, 1)
    learning_rate_decay = float_input('learning decay rate (suggested: 0.000001)', 0, 1)
    cycle = int_input('number of cycles before resetting the learning rate (suggested: 3)', 1, 10)
    num_rounds = int_input('number of rounds (suggested: 3)', 1, 100)
    print('Select image resolution:')
    print('[0] low (' + str(img_dim) + ' px)')
    print('[1] mid (' + str(img_dim * 2) + ' px)')
    print('[2] high (' + str(img_dim * 4) + ' px)')
    img_resolution_index = int_input('choice', 0, 2, show_range = False)
    if img_resolution_index == 0:
        img_size = 1
    elif img_resolution_index == 1:
        img_size = 2
    else:
        img_size = 4
    use_augmentation = str_input('Would you like to add image augmentation? (yes or no) ', ['yes', 'no'])
    if use_augmentation == 'yes':
        augmentations = select_augmentations()
    else:
        augmentations = None

    project = {'name': project_name,
               'img_path': image_path,
               'path': project_path,
               'plot': plot_cm,
               'api_port': api_port,
               'kfold': kfold,
               'kfold_every': kfold_every,
               'cycle': cycle,
               'seed': np.random.randint(9999),
               'batch_size': batch_size,
               'learning_rate': learning_rate,
               'learning_rate_decay': learning_rate_decay,
               'final_cutoff': final_cutoff,
               'rounds': num_rounds,
               'img_size': img_size,
               'augmentations': augmentations,
               'architecture': arch,
               'img_dim': img_dim,
               'conv_dim': conv_dim,
               'is_split': False,
               'is_array': False,
               'is_augmented': False,
               'is_pre_model': False,
               'is_final': False,
               'model_round': 0,
               'server_weights': None,
               'last_weights': None,
               'best_weights': None}

    config.append(project)
    store_config(config)
    print('')
    print(colored('Project configure saved!', 'cyan'))
    print('')
    print('To run project:')
    print('')
    print(colored('    transfer --run --project ' + project_name, 'green'))
    print('or')
    print(colored('    transfer -r -p ' + project_name, 'green'))