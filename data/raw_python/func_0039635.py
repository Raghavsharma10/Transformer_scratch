def to_superuser(email_class, **data):
    """
    Email superusers
    """
    for user in get_user_model().objects.filter(is_superuser=True):
        try:
            email_class().send([user.email], user.language, **data)
        except AttributeError:
            email_class().send([user.email], translation.get_language(), **data)