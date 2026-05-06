def find_page_location(command, specified_platform):
    """Find the command man page in the pages directory."""
    repo_directory = get_config()['repo_directory']
    default_platform = get_config()['platform']
    command_platform = (
        specified_platform if specified_platform else default_platform)

    with io.open(path.join(repo_directory, 'pages/index.json'),
                 encoding='utf-8') as f:
        index = json.load(f)
    command_list = [item['name'] for item in index['commands']]
    if command not in command_list:
        sys.exit(
            ("Sorry, we don't support command: {0} right now.\n"
             "You can file an issue or send a PR on github:\n"
             "    https://github.com/tldr-pages/tldr").format(command))

    supported_platforms = index['commands'][
        command_list.index(command)]['platform']
    if command_platform in supported_platforms:
        platform = command_platform
    elif 'common' in supported_platforms:
        platform = 'common'
    else:
        platform = ''
    if not platform:
        sys.exit(
            ("Sorry, command {0} is not supported on your platform.\n"
             "You can file an issue or send a PR on github:\n"
             "    https://github.com/tldr-pages/tldr").format(command))

    page_path = path.join(path.join(repo_directory, 'pages'),
                          path.join(platform, command + '.md'))
    return page_path