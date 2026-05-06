def build_url(component, filename, **values):
    """
    search bower asset and build url

    :param component: bower component (package)
    :type component: str
    :param filename: filename in bower component - can contain directories (like dist/jquery.js)
    :type filename: str
    :param values: additional url parameters
    :type values: dict[str, str]
    :return: url
    :rtype: str | None
    """
    root = current_app.config['BOWER_COMPONENTS_ROOT']
    bower_data = None
    package_data = None

    # check if component exists in bower_components directory
    if not os.path.isdir(os.path.join(current_app.root_path, root, component)):
        # FallBack to default url_for flask
        return None

    # load bower.json of specified component
    bower_file_path = os.path.join(current_app.root_path, root, component, 'bower.json')
    if os.path.exists(bower_file_path):
        with open(bower_file_path, 'r') as bower_file:
            bower_data = json.load(bower_file)

    # check if package.json exists and load package.json data
    package_file_path = os.path.join(current_app.root_path, root, component, 'package.json')
    if os.path.exists(package_file_path):
        with open(package_file_path, 'r') as package_file:
            package_data = json.load(package_file)

    # check if specified file actually exists
    if not os.path.exists(os.path.join(current_app.root_path, root, component, filename)):
        return None

    # check if minified file exists (by pattern <filename>.min.<ext>
    # returns filename if successful
    if current_app.config['BOWER_TRY_MINIFIED']:
        if '.min.' not in filename:
            minified_filename = '%s.min.%s' % tuple(filename.rsplit('.', 1))
            minified_path = os.path.join(root, component, minified_filename)

            if os.path.exists(os.path.join(current_app.root_path, minified_path)):
                filename = minified_filename

    # determine version of component and append as ?version= parameter to allow cache busting
    if current_app.config['BOWER_QUERYSTRING_REVVING']:
        if bower_data is not None and 'version' in bower_data:
            values['version'] = bower_data['version']
        elif package_data is not None and 'version' in package_data:
            values['version'] = package_data['version']
        else:
            values['version'] = os.path.getmtime(os.path.join(current_app.root_path, root, component, filename))

    return url_for('bower.serve', component=component, filename=filename, **values)