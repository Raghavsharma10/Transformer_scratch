def combine_or(matcher, *more_matchers):
    """Combines more than one matcher together (first that matches wins)."""

    def matcher(cause):
        for sub_matcher in itertools.chain([matcher], more_matchers):
            cause_cls = sub_matcher(cause)
            if cause_cls is not None:
                return cause_cls
        return None

    return matcher