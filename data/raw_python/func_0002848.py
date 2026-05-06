def get_projects(osa_repo_dir, commit):
    """Get all projects from multiple YAML files."""
    # Check out the correct commit SHA from the repository
    repo = Repo(osa_repo_dir)
    checkout(repo, commit)

    yaml_files = glob.glob(
        '{0}/playbooks/defaults/repo_packages/*.yml'.format(osa_repo_dir)
    )
    yaml_parsed = []
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            yaml_parsed.append(yaml.load(f))

    merged_dicts = {k: v for d in yaml_parsed for k, v in d.items()}

    return normalize_yaml(merged_dicts)