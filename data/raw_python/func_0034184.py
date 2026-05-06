def create(request):
    """Create a new poll"""

    errors = []
    success = False
    listOfResponses = ['', '', '']  # 3 Blank lines by default
    title = ''
    description = ''
    id = ''

    if request.method == 'POST':  # User saved the form
        # Retrieve parameters
        title = request.form.get('title')
        description = request.form.get('description')

        listOfResponses = []
        for rep in request.form.getlist('rep[]'):
            if rep != '':
                listOfResponses.append(rep)

        # Test if everything is ok
        if title == "":
            errors.append("Please set a title !")

        if len(listOfResponses) == 0:
            errors.append("Please set at least one response !")

        # Can we save the new question ?
        if len(errors) == 0:
            # Yes. Let save data
            curDB.execute("INSERT INTO Poll (title, description) VALUES (?, ?)", (title, description))

            # The id of the poll
            id = curDB.lastrowid

            # Insert responses
            for rep in listOfResponses:
                curDB.execute("INSERT INTO Response (pollId, title) VALUES (?, ?)", (id, rep))

            coxDB.commit()

            success = True

        # Minimum of 3 lines of questions
        while len(listOfResponses) < 3:
            listOfResponses.append('')

    return {'errors': errors, 'success': success, 'listOfResponses': listOfResponses, 'title': title, 'description': description, 'id': id}