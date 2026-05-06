def user_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Sphinx role for linking to a user profile. Defaults to linking to
    Github profiles, but the profile URIS can be configured via the
    ``issues_user_uri`` config value.

    Examples: ::

        :user:`sloria`

    Anchor text also works: ::

        :user:`Steven Loria <sloria>`
    """
    options = options or {}
    content = content or []
    has_explicit_title, title, target = split_explicit_title(text)

    target = utils.unescape(target).strip()
    title = utils.unescape(title).strip()
    config = inliner.document.settings.env.app.config
    if config.issues_user_uri:
        ref = config.issues_user_uri.format(user=target)
    else:
        ref = "https://github.com/{0}".format(target)
    if has_explicit_title:
        text = title
    else:
        text = "@{0}".format(target)

    link = nodes.reference(text=text, refuri=ref, **options)
    return [link], []