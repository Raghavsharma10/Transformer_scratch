def _get_assistants_snippets(path, name):
    '''Get Assistants and Snippets for a given DAP name on a given path'''
    result = []
    subdirs = {'assistants': 2, 'snippets': 1} # Values used for stripping leading path tokens

    for loc in subdirs:
        for root, dirs, files in os.walk(os.path.join(path, loc)):
            for filename in [utils.strip_prefix(os.path.join(root, f), path) for f in files]:
                stripped = os.path.sep.join(filename.split(os.path.sep)[subdirs[loc]:])
                if stripped.startswith(os.path.join(name, '')) or stripped == name + '.yaml':
                    result.append(os.path.join('fakeroot', filename))

    return result