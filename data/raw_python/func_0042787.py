def get_game(site, description="", create=False):
    """
        get the current game, if its still active, else
        creates a new game, if the current time is inside the
        GAME_START_TIMES interval and create=True
        @param create: create a game, if there is no active game
        @returns: None if there is no active Game, and none shoul be
        created or the (new) active Game.
    """

    game = None
    games = Game.objects.filter(site=site).order_by("-created")
    try:
        game = games[0]
    except IndexError:
        game = None

    # no game, yet, or game expired
    if game is None or game.is_expired() or is_after_endtime():
        if create:
            if is_starttime():
                game = Game(site=site, description=description)
                game.save()
            else:
                raise TimeRangeError(
                    _(u"game start outside of the valid timerange"))
        else:
            game = None

    # game exists and its not after the GAME_END_TIME
    elif not is_after_endtime():
        game = games[0]

    return game