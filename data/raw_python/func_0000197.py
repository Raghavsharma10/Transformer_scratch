def load_env(file):
    """
    Generate environment used for 'org.restore' method
    :param file: env file
    :return: env
    """

    env = yaml.load(open(file))

    for org in env.get('organizations', []):
        if not org.get('applications'):
            org['applications'] = []

        if org.get('starter-kit'):
            kit_meta = get_starter_kit_meta(org.get('starter-kit'))
            for meta_app in get_applications_from_metadata(kit_meta):
                org['applications'].append(meta_app)

        if org.get('meta'):
            for meta_app in get_applications_from_metadata(org.get('meta')):
                org['applications'].append(meta_app)

        for app in org.get('applications', []):
            if app.get('file'):
                app['file'] = os.path.realpath(os.path.join(os.path.dirname(file), app['file']))
    return env