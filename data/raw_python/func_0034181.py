def home(request):
    """Show the home page. Send the list of polls"""

    polls = []

    for row in curDB.execute('SELECT id, title FROM Poll ORDER BY title'):
        polls.append({'id': row[0], 'name': row[1]})

    return {'polls': polls}