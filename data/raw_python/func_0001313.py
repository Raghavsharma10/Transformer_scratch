def create_admin(user_config_path: str = 'CONFIG.superuser') -> bool:
    """
    Creates a superuser from a specified dict/object bundle located at ``user_config_path``.
    Skips if the specified object contains no email or no username.
    If a user with the specified username already exists and has no usable password it updates user's password with
    a specified one.

    ``user_config_path`` can accept any path to a deep nested object, like dict of dicts,
    object of dicts of objects, and so on. Let's assume you have this weird config in your ``settings.py``:
    ::

        class MyConfigObject:
            my_var = {
                'user': {
                    'username': 'user',
                    'password': 'qwe',
                    'email': 'no@example.com',
                }
            }
        local_config = MyConfigObject()

    To access the ``'user'`` bundle you have to specify: ``local_config.my_var.user``.

    :param user_config_path: dot-separated path to object or dict, default is ``'CONFIG.superuser'``
    :return: ``True`` if user has been created, ``False`` otherwise
    """
    from django.conf import settings
    wf('Creating superuser... ', False)
    username, email, password = [
        dot_path(settings, '{0}.{1}'.format(user_config_path, 'username')),
        dot_path(settings, '{0}.{1}'.format(user_config_path, 'email')),
        dot_path(settings, '{0}.{1}'.format(user_config_path, 'password')),
    ]
    if not all([username, email]):
        wf('[SKIP: username and email should not be empty]\n')
        return False

    from django.db import IntegrityError
    try:
        execute_from_command_line([
            './manage.py', 'createsuperuser',
            '--username', username,
            '--email', email,
            '--noinput'
        ])
    except IntegrityError:
        pass

    if password:
        # after `execute_from_command_line` models are loaded
        from django.contrib.auth.models import User
        user = User.objects.get(username=username)

        # do not change password if it was set before
        if not user.has_usable_password():
            user.set_password(password)
            user.save()
    else:
        wf('[SKIP update password: password is empty]\n')

    wf('[+]\n')
    return True