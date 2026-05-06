def _post_vote(user_bingo_board, field, vote):
    """
        change vote on a field
        @param user_bingo_board: the user's bingo board or None
        @param field: the BingoField to vote on
        @param vote: the vote property from the HTTP POST
        @raises: VoteException: if user_bingo_board is None or
        the field does not belong to the user's bingo board.
        @raises Http404: If the BingoField does not exist.
    """

    if field.board != user_bingo_board:
        raise VoteException(
            "the voted field does not belong to the user's BingoBoard")

    if vote == "0":
        field.vote = 0
    elif vote == "+":
        field.vote = +1
    elif vote == "-":
        field.vote = -1
    field.save()

    # update last_used with current timestamp
    game = user_bingo_board.game
    Game.objects.filter(id=game.id).update(
        last_used=times.now())

    # invalidate vote cache
    vote_counts_cachename = 'vote_counts_game={0:d}'.format(
        field.board.game.id)
    cache.delete(vote_counts_cachename)

    # publish the new vote counts for server-sent events
    if USE_SSE:
        try:
            votes = field.num_votes()
            redis.publish("word_votes", json.dumps({
                'site_id': game.site.id,
                'word_id': field.word.id,
                'vote_count': votes,
            }))
            redis.publish("field_vote", json.dumps({
                'site_id': game.site.id,
                'field_id': field.id,
                'vote': vote,
            }))
        except RedisConnectionError:
            # redis server not available?
            pass