def cve_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Sphinx role for linking to a CVE on https://cve.mitre.org.

    Examples: ::

        :cve:`CVE-2018-17175`

    """
    options = options or {}
    content = content or []
    has_explicit_title, title, target = split_explicit_title(text)

    target = utils.unescape(target).strip()
    title = utils.unescape(title).strip()
    ref = "https://cve.mitre.org/cgi-bin/cvename.cgi?name={0}".format(target)
    text = title if has_explicit_title else target
    link = nodes.reference(text=text, refuri=ref, **options)
    return [link], []