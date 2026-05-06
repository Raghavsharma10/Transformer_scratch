def extract_apps(image, app_names):
    ''' extract app will extract metadata for one or more apps
        Parameters
        ==========
        image: the absolute path to the image
        app_name: the name of the app under /scif/apps

    '''
    apps = dict()

    if isinstance(app_names, tuple):
        app_names = list(app_names)
    if not isinstance(app_names, list):
        app_names = [app_names]
    if len(app_names) == 0:
        return apps

    for app_name in app_names:
        metadata = dict()

        # Inspect: labels, env, runscript, tests, help
        try:
            inspection = json.loads(Client.inspect(image, app=app_name))
            del inspection['data']['attributes']['deffile']
            metadata['inspect'] = inspection

        # If illegal characters prevent load, not much we can do
        except:
            pass
        apps[app_name] = metadata
    return apps