def contacts(github, logins):
    """Extract public contact info from users.
    """
    printmp('Fetching contacts')
    users = [github.user(login).as_dict() for login in logins]
    mails = set()
    blogs = set()
    for user in users:
        contact = user.get('name', 'login')
        if user['email']:
            contact += ' <%s>' % user['email']
            mails.add(contact)
        elif user['blog']:
            contact += ' <%s>' % user['blog']
            blogs.add(contact)
        else:
            continue
    return mails, blogs