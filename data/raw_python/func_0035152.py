def create_settings(pkg, repo_dest, db_user, db_name, db_password, db_host,
                    db_port):
    """
    Creates a local settings file out of the distributed template.

    This also fills in database settings and generates a secret key, etc.
    """
    vars = {'pkg': pkg,
            'db_user': db_user,
            'db_name': db_name,
            'db_password': db_password or '',
            'db_host': db_host or '',
            'db_port': db_port or '',
            'hmac_date': datetime.now().strftime('%Y-%m-%d'),
            'hmac_key': generate_key(32),
            'secret_key': generate_key(32)}
    with dir_path(repo_dest):
        shutil.copyfile('%s/settings/local.py-dist' % pkg,
                        '%s/settings/local.py' % pkg)
        patch("""\
            --- a/%(pkg)s/settings/local.py
            +++ b/%(pkg)s/settings/local.py
            @@ -9,11 +9,11 @@ from . import base
             DATABASES = {
                 'default': {
                     'ENGINE': 'django.db.backends.mysql',
            -        'NAME': 'playdoh_app',
            -        'USER': 'root',
            -        'PASSWORD': '',
            -        'HOST': '',
            -        'PORT': '',
            +        'NAME': '%(db_name)s',
            +        'USER': '%(db_user)s',
            +        'PASSWORD': '%(db_password)s',
            +        'HOST': '%(db_host)s',
            +        'PORT': '%(db_port)s',
                     'OPTIONS': {
                         'init_command': 'SET storage_engine=InnoDB',
                         'charset' : 'utf8',
            @@ -51,14 +51,14 @@ DEV = True
             # Playdoh ships with Bcrypt+HMAC by default because it's the most secure.
             # To use bcrypt, fill in a secret HMAC key. It cannot be blank.
             HMAC_KEYS = {
            -    #'2012-06-06': 'some secret',
            +    '%(hmac_date)s': '%(hmac_key)s',
             }

             from django_sha2 import get_password_hashers
             PASSWORD_HASHERS = get_password_hashers(base.BASE_PASSWORD_HASHERS, HMAC_KEYS)

             # Make this unique, and don't share it with anybody.  It cannot be blank.
            -SECRET_KEY = ''
            +SECRET_KEY = '%(secret_key)s'

             # Uncomment these to activate and customize Celery:
             # CELERY_ALWAYS_EAGER = False  # required to activate celeryd
            """ % vars)