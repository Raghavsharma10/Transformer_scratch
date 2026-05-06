def to_user(email_class, user, **data):
    """
    Email user
    """
    try:
        email_class().send([user.email], user.language, **data)
    except AttributeError:
        # this is a fallback in case the user model does not have the language field
        email_class().send([user.email], translation.get_language(), **data)