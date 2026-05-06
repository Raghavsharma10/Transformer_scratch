def download_api(branch=None) -> str:
    """download API documentation from _branch_ of Habitica\'s repo on Github"""
    habitica_github_api = 'https://api.github.com/repos/HabitRPG/habitica'
    if not branch:
        branch = requests.get(habitica_github_api + '/releases/latest').json()['tag_name']
    curl = local['curl']['-sL', habitica_github_api + '/tarball/{}'.format(branch)]
    tar = local['tar'][
        'axzf', '-', '--wildcards', '*/website/server/controllers/api-v3/*', '--to-stdout']
    grep = local['grep']['@api']
    sed = local['sed']['-e', 's/^[ */]*//g', '-e', 's/  / /g', '-']
    return (curl | tar | grep | sed)()