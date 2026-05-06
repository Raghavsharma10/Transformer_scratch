def handle(app):
    # TODO:  for this to work properly we need a generator registry
    # generator, lifecycle etc.
    # list of tuples (label, value)
    # TODO customize & use own style

    default_choices = [
        {
            'name': 'Install a generator',
            'value': 'install'
        },
        {
            'name': 'Find some help',
            'value': 'help'
        },
        {
            'name': 'Get me out of here!',
            'value': 'exit'
        }
    ]

    '''
    if (globalConfigHasContent()) {
    defaultChoices.splice(defaultChoices.length - 1, 0, {
      name: 'Clear global config',
      value: 'clearConfig'
    });
    }

    var generatorList = _.chain(app.generators).map(function (generator) {
    if (!generator.appGenerator) {
      return null;
    }

    var updateInfo = generator.updateAvailable ? chalk.dim.yellow(' ♥ Update Available!') : '';

    return {
      name: generator.prettyName + updateInfo,
      value: {
        method: 'run',
        generator: generator.namespace
      }
    };
    }).compact().sortBy(function (el) {
    var generatorName = namespaceToName(el.value.generator);
    return -app.conf.get('generatorRunCount')[generatorName] || 0;
    }).value();

    if (generatorList.length) {
    defaultChoices.unshift({
      name: 'Update your generators',
      value: 'update'
    });
    }
    '''

    # app.insight.track('yoyo', 'home');
    generator_list = [{'name': g.title(), 'value': {'name': g, 'method': 'run'}}
                      for g in app.generators]

    choices = _flatten([
        whaaaaat.Separator('Run a generator'),
        generator_list,
        whaaaaat.Separator(),
        default_choices,
        whaaaaat.Separator(),
    ])

    # var allo = name ? '\'Allo ' + name.split(' ')[0] + '! ' : '\'Allo! ';
    allo = 'MoinMoin! '

    questions = [
        {
            'type': 'list',
            'name': 'what_next',
            'message': allo + 'What would you like to do?',
            'choices': choices,
        }
    ]

    answer = whaaaaat.prompt(questions)

    if isinstance(answer['what_next'], dict) and \
            answer['what_next']['method'] == 'run':
        app.navigate('run', answer['what_next']['name'])
        return
    elif answer['what_next'] == 'exit':
        return

    app.navigate(answer['what_next'])