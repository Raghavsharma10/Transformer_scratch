def rule_factory(plural, singular):
    """Element rule factory."""
    @rules.rule(plural)
    def f(path, values):
        for v in values:
            if v:
                elem = etree.Element(
                    '{{http://purl.org/dc/elements/1.1/}}{0}'.format(singular))
                elem.text = v
                yield elem
    f.__name__ = plural
    return f