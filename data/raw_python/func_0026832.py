def authorized(self, environ):
        """
        If we're running Django and ``GNOTTY_LOGIN_REQUIRED`` is set
        to ``True``, pull the session cookie from the environment and
        validate that the user is authenticated.
        """
        if self.django and settings.LOGIN_REQUIRED:
            try:
                from django.conf import settings as django_settings
                from django.contrib.auth import SESSION_KEY
                from django.contrib.auth.models import User
                from django.contrib.sessions.models import Session
                from django.core.exceptions import ObjectDoesNotExist
                cookie = SimpleCookie(environ["HTTP_COOKIE"])
                cookie_name = django_settings.SESSION_COOKIE_NAME
                session_key = cookie[cookie_name].value
                session = Session.objects.get(session_key=session_key)
                user_id = session.get_decoded().get(SESSION_KEY)
                user = User.objects.get(id=user_id)
            except (ImportError, KeyError, ObjectDoesNotExist):
                return False
        return True