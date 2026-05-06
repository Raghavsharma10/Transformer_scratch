def get_current_version(repo_path):
    """
    Given a repo will return the version string, according to semantic
    versioning, counting as non-backwards compatible commit any one with a
    message header that matches (case insensitive)::

        sem-ver: .*break.*

    And as features any commit with a header matching::

        sem-ver: feature

    And counting any other as a bugfix
    """
    repo = dulwich.repo.Repo(repo_path)
    tags = get_tags(repo)
    maj_version = 0
    feat_version = 0
    fix_version = 0

    for commit_sha, children in reversed(
            get_children_per_first_parent(repo_path).items()
    ):
        commit = get_repo_object(repo, commit_sha)
        maj_version, feat_version, fix_version = get_version(
            commit=commit,
            tags=tags,
            maj_version=maj_version,
            feat_version=feat_version,
            fix_version=fix_version,
            children=children,
        )

    return '%s.%s.%s' % (maj_version, feat_version, fix_version)