def check_auth(user):
    '''
    Check if the user should or shouldn't be inside the system:
    - If the user is staff or superuser: LOGIN GRANTED
    - If the user has a Person and it is not "disabled": LOGIN GRANTED
    - Elsewhere: LOGIN DENIED
    '''

    # Initialize authentication
    auth = None
    person = None

    # Check if there is an user
    if user:

        # It means that Django accepted the user and it is active
        if user.is_staff or user.is_superuser:
            # This is an administrator, let it in
            auth = user
        else:
            # It is a normal user, check if there is a person behind
            person = getattr(user, "person", None)
            if not person:
                # Check if there is related one
                person_related = getattr(user, "people", None)
                if person_related:
                    # Must be only one
                    if person_related.count() == 1:
                        person = person_related.get()

            if person and ((person.disabled is None) or (person.disabled > timezone.now())):
                # There is a person, no disabled found or the found one is fine to log in
                auth = user

    # Return back the final decision
    return auth