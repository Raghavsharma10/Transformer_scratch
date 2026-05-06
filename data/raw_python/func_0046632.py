def upload_settings():
    "Uploads the non-versioned local settings to the server."
    local_path = os.path.join(curdir, 'settings/{0}.py'.format(env.host))
    if os.path.exists(local_path):
        remote_path = os.path.join(env.path, 'varify/conf/local_settings.py')
        put(local_path, remote_path)
    elif not confirm(yellow('No local settings found for host "{0}". Continue anyway?'.format(env.host))):
        abort('No local settings found for host "{0}". Aborting.'.format(env.host))