def get_authors(repo_path, from_commit):
    """
    Given a repo and optionally a base revision to start from, will return
    the list of authors.
    """
    repo = dulwich.repo.Repo(repo_path)
    refs = get_refs(repo)
    start_including = False
    authors = set()

    if from_commit is None:
        start_including = True

    for commit_sha, children in reversed(
        get_children_per_first_parent(repo_path).items()
    ):
        commit = get_repo_object(repo, commit_sha)
        if (
            start_including or commit_sha.startswith(from_commit) or
            fuzzy_matches_refs(from_commit, refs.get(commit_sha, []))
        ):
            authors.add(commit.author.decode())
            for child in children:
                authors.add(child.author.decode())

            start_including = True

    return '\n'.join(sorted(authors))