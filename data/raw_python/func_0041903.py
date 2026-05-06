def extract_mail(issues):
    """Extract mails that sometimes leak from issue comments.
    """
    contacts = set()
    for idx, issue in enumerate(issues):
        printmp('Fetching issue #%s' % idx)
        for comment in issue.comments():
            comm = comment.as_dict()
            emails = list(email[0] for email in re.findall(MAIL_REGEX, comm['body'])
                if not email[0].startswith('//') and not email[0].endswith('github.com') and
                '@' in email[0])
            contacts |= set(emails)
    return contacts