def _ask_questionnaire():
    """Asks questions to fill out a HFOS plugin template"""

    answers = {}
    print(info_header)
    pprint(questions.items())

    for question, default in questions.items():
        response = _ask(question, default, str(type(default)), show_hint=True)
        if type(default) == unicode and type(response) != str:
            response = response.decode('utf-8')
        answers[question] = response

    return answers