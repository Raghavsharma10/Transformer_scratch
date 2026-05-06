def to_staff(email_class, **data):
    """
    Email staff users
    """
    for user in get_user_model().objects.filter(is_staff=True):
        try:
            email_class().send([user.email], user.language, **data)
        except AttributeError:
            email_class().send([user.email], translation.get_language(), **data)