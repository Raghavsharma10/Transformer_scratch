def make_osa_report(repo_dir, old_commit, new_commit,
                    args):
    """Create initial RST report header for OpenStack-Ansible."""
    update_repo(repo_dir, args.osa_repo_url, args.update)

    # Are these commits valid?
    validate_commits(repo_dir, [old_commit, new_commit])

    # Do we have a valid commit range?
    validate_commit_range(repo_dir, old_commit, new_commit)

    # Get the commits in the range
    commits = get_commits(repo_dir, old_commit, new_commit)

    # Start off our report with a header and our OpenStack-Ansible commits.
    template_vars = {
        'args': args,
        'repo': 'openstack-ansible',
        'commits': commits,
        'commit_base_url': get_commit_url(args.osa_repo_url),
        'old_sha': old_commit,
        'new_sha': new_commit
    }
    return render_template('offline-header.j2', template_vars)