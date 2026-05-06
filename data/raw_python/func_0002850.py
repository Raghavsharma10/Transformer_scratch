def get_roles(osa_repo_dir, commit, role_requirements):
    """Read OSA role information at a particular commit."""
    repo = Repo(osa_repo_dir)

    checkout(repo, commit)

    log.info("Looking for file {f} in repo {r}".format(r=osa_repo_dir,
                                                       f=role_requirements))
    filename = "{0}/{1}".format(osa_repo_dir, role_requirements)
    with open(filename, 'r') as f:
        roles_yaml = yaml.load(f)

    return normalize_yaml(roles_yaml)