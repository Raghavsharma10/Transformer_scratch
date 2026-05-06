def ckan_extension_template(name, target):
    """
    Create ckanext-(name) in target directory.
    """
    setupdir = '{0}/ckanext-{1}theme'.format(target, name)
    extdir = setupdir + '/ckanext/{0}theme'.format(name)
    templatedir = extdir + '/templates/'
    staticdir = extdir + '/static/datacats'

    makedirs(templatedir + '/home/snippets')
    makedirs(staticdir)

    here = dirname(__file__)
    copyfile(here + '/images/chart.png', staticdir + '/chart.png')
    copyfile(here + '/images/datacats-footer.png',
        staticdir + '/datacats-footer.png')

    filecontents = [
        (setupdir + '/setup.py', SETUP_PY),
        (setupdir + '/.gitignore', DOT_GITIGNORE),
        (setupdir + '/ckanext/__init__.py', NAMESPACE_PACKAGE),
        (extdir + '/__init__.py', ''),
        (extdir + '/plugins.py', PLUGINS_PY),
        (templatedir + '/home/snippets/promoted.html', PROMOTED_SNIPPET),
        (templatedir + '/footer.html', FOOTER_HTML),
        ]

    for filename, content in filecontents:
        with open(filename, 'w') as f:
            f.write(content.replace('##name##', name))