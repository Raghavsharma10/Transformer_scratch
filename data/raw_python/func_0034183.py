def vote(request, pollId, responseId):
    """Vote for a poll"""

    username = request.args.get('ebuio_u_username')

    # Remove old votes from the same user on the same poll
    curDB.execute('DELETE FROM Vote WHERE username = ?  AND responseId IN (SELECT id FROM Response WHERE pollId = ?) ', (username, pollId))

    # Save the vote
    curDB.execute('INSERT INTO Vote (username, responseID) VALUES (?, ?) ', (username, responseId))

    coxDB.commit()

    # To display a success page, comment this line
    return PlugItRedirect("show/" + str(pollId))

    # Line to display the success page
    return {'id': pollId}