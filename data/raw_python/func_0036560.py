def get_user_model():
    """
    Returns the user model to use at runtime.
    :return: User or custom user
    """
    if DJANGO_VERSION >= (1, 5):
        from django.contrib.auth import get_user_model

        return get_user_model()  # NOQA
    else:
        from django.contrib.auth.models import User  # NOQA
        return User