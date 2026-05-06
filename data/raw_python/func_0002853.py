def normalize_yaml(yaml):
    """Normalize the YAML from project and role lookups.

    These are returned as a list of tuples.
    """
    if isinstance(yaml, list):
        # Normalize the roles YAML data
        normalized_yaml = [(x['name'], x['src'], x.get('version', 'HEAD'))
                           for x in yaml]
    else:
        # Extract the project names from the roles YAML and create a list of
        # tuples.
        projects = [x[:-9] for x in yaml.keys() if x.endswith('git_repo')]
        normalized_yaml = []
        for project in projects:
            repo_url = yaml['{0}_git_repo'.format(project)]
            commit_sha = yaml['{0}_git_install_branch'.format(project)]
            normalized_yaml.append((project, repo_url, commit_sha))

    return normalized_yaml