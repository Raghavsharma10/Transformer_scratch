def get_authuser_model():
    """ Define and return AuthUser model using nefertari base classes """
    from nefertari.authentication.models import AuthUserMixin
    from nefertari import engine

    class AuthUser(AuthUserMixin, engine.BaseDocument):
        __tablename__ = 'ramses_authuser'

    return AuthUser