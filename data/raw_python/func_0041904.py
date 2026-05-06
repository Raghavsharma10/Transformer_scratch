def fetch_logins(roles, repo):
    """Fetch logins for users with given roles.
    """
    users = set()
    if 'stargazer' in roles:
        printmp('Fetching stargazers')
        users |= set(repo.stargazers())
    if 'collaborator' in roles:
        printmp('Fetching collaborators')
        users |= set(repo.collaborators())
    if 'issue' in roles:
        printmp('Fetching issues creators')
        users |= set([i.user for i in repo.issues(state='all')])
    return users