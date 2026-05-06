def _find_supported_challenge(authzr, responders):
    """
    Find a challenge combination that consists of a single challenge that the
    responder can satisfy.

    :param ~acme.messages.AuthorizationResource auth: The authorization to
        examine.

    :type responder: List[`~txacme.interfaces.IResponder`]
    :param responder: The possible responders to use.

    :raises NoSupportedChallenges: When a suitable challenge combination is not
        found.

    :rtype: Tuple[`~txacme.interfaces.IResponder`,
            `~acme.messages.ChallengeBody`]
    :return: The responder and challenge that were found.
    """
    matches = [
        (responder, challbs[0])
        for challbs in authzr.body.resolved_combinations
        for responder in responders
        if [challb.typ for challb in challbs] == [responder.challenge_type]]
    if len(matches) == 0:
        raise NoSupportedChallenges(authzr)
    else:
        return matches[0]