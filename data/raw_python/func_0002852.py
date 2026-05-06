def make_report(storage_directory, old_pins, new_pins, do_update=False,
                version_mappings=None):
    """Create RST report from a list of projects/roles."""
    report = ""
    version_mappings = version_mappings or {}
    for new_pin in new_pins:
        repo_name, repo_url, commit_sha = new_pin
        commit_sha = version_mappings.get(repo_name, {}
                                          ).get(commit_sha, commit_sha)

        # Prepare our repo directory and clone the repo if needed. Only pull
        # if the user requests it.
        repo_dir = "{0}/{1}".format(storage_directory, repo_name)
        update_repo(repo_dir, repo_url, do_update)

        # Get the old SHA from the previous pins. If this pin didn't exist
        # in the previous OSA revision, skip it. This could happen with newly-
        # added projects and roles.
        try:
            commit_sha_old = next(x[2] for x in old_pins if x[0] == repo_name)
        except Exception:
            continue
        else:
            commit_sha_old = version_mappings.get(repo_name, {}
                                                  ).get(commit_sha_old,
                                                        commit_sha_old)

        # Loop through the commits and render our template.
        validate_commits(repo_dir, [commit_sha_old, commit_sha])
        commits = get_commits(repo_dir, commit_sha_old, commit_sha)
        template_vars = {
            'repo': repo_name,
            'commits': commits,
            'commit_base_url': get_commit_url(repo_url),
            'old_sha': commit_sha_old,
            'new_sha': commit_sha
        }
        rst = render_template('offline-repo-changes.j2', template_vars)
        report += rst

    return report