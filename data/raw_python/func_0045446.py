def solveAndNotify(proto, exercise):
    """The user at the given AMP protocol has solved the given exercise.

    This will log the solution and notify the user.
    """
    exercise.solvedBy(proto.user)
    proto.callRemote(ce.NotifySolved,
                     identifier=exercise.identifier,
                     title=exercise.title)