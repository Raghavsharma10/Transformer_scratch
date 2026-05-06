def show(request, pollId):
    """Show a poll. We send informations about votes only if the user is an administrator"""

    # Get the poll
    curDB.execute('SELECT id, title, description FROM Poll WHERE id = ? ORDER BY title', (pollId,))
    poll = curDB.fetchone()

    if poll is None:
        return {}

    responses = []
    totalVotes = 0
    votedFor = 0

    # Compute the list of responses
    curDB.execute('SELECT id, title FROM Response WHERE pollId = ? ORDER BY title ', (poll[0],))
    resps = curDB.fetchall()

    for rowRep in resps:

        votes = []
        nbVotes = 0

        # List each votes
        for rowVote in curDB.execute('SELECT username FROM Vote WHERE responseId = ?', (rowRep[0],)):

            nbVotes += 1

            # If the user is and administrator, saves each votes
            if request.args.get('ebuio_u_ebuio_admin') == 'True':
                votes.append(rowVote[0])

            # Save the vote of the current suer
            if request.args.get('ebuio_u_username') == rowVote[0]:
                votedFor = rowRep[0]

        totalVotes += nbVotes
        responses.append({'id': rowRep[0], 'title': rowRep[1], 'nbVotes': nbVotes, 'votes': votes})

    return {'id': poll[0], 'name': poll[1], 'description': poll[2], 'responses': responses, 'totalVotes': totalVotes, 'votedFor': votedFor}