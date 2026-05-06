def update_session(request, session_to_set, hproPk):
    """Update the session with users-realted values"""

    for key, value in session_to_set.items():
        request.session['plugit_' + str(hproPk) + '_' + key] = value