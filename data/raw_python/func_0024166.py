def set_password(sender, **kwargs):
    """
    Encrypts password of the user.
    """
    if sender.model_class.__name__ == 'User':
        usr = kwargs['object']
        if not usr.password.startswith('$pbkdf2'):
            usr.set_password(usr.password)
            usr.save()