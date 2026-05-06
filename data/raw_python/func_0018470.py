def init():
    """Init config file."""
    default_config_path = path.join(
        (os.environ.get('TLDR_CONFIG_DIR') or path.expanduser('~')),
        '.tldrrc')
    if path.exists(default_config_path):
        click.echo("There is already a config file exists, "
                   "skip initializing it.")
    else:
        repo_path = click.prompt("Input the tldr repo path(absolute path)")
        if not path.exists(repo_path):
            sys.exit("Repo path not exist, clone it first.")

        platform = click.prompt("Input your platform(linux, osx or sunos)")
        if platform not in ['linux', 'osx', 'sunos']:
            sys.exit("Platform should be in linux, osx or sunos.")

        colors = {
            "description": "blue",
            "usage": "green",
            "command": "cyan"
        }

        config = {
            "repo_directory": repo_path,
            "colors": colors,
            "platform": platform
        }
        with open(default_config_path, 'w') as f:
            f.write(yaml.safe_dump(config, default_flow_style=False))

        click.echo("Initializing the config file at {0}".format(
            default_config_path))