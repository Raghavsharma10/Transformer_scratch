def high_cli(repo_name, login, with_blog, as_list, role):
    """Extract mails from stargazers, collaborators and people involved with issues of given
    repository.
    """
    passw = getpass.getpass()
    github = gh_login(login, passw)
    repo = github.repository(login, repo_name)
    role = [ROLES[k] for k in role]
    users = fetch_logins(role, repo)
    mails, blogs = contacts(github, users)

    if 'issue' in role:
        mails |= extract_mail(repo.issues(state='all'))

    # Print results
    sep = ', ' if as_list else '\n'
    print(sep.join(mails))
    if with_blog:
        print(sep.join(blogs))