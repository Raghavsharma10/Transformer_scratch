def generate_user(mode=None, pk=None):
    """Return a false user for standalone mode"""

    user = None

    if mode == 'log' or pk == "-1":
        user = DUser(pk=-1, username='Logged', first_name='Logged', last_name='Hector', email='logeedin@plugit-standalone.ebuio')
        user.gravatar = 'https://www.gravatar.com/avatar/ebuio1?d=retro'
        user.ebuio_member = False
        user.ebuio_admin = False
        user.subscription_labels = []
    elif mode == 'mem' or pk == "-2":
        user = DUser(pk=-2, username='Member', first_name='Member', last_name='Luc', email='memeber@plugit-standalone.ebuio')
        user.gravatar = 'https://www.gravatar.com/avatar/ebuio2?d=retro'
        user.ebuio_member = True
        user.ebuio_admin = False
        user.subscription_labels = []
    elif mode == 'adm' or pk == "-3":
        user = DUser(pk=-3, username='Admin', first_name='Admin', last_name='Charles', email='admin@plugit-standalone.ebuio')
        user.gravatar = 'https://www.gravatar.com/avatar/ebuio3?d=retro'
        user.ebuio_member = True
        user.ebuio_admin = True
        user.subscription_labels = []
    elif mode == 'ano':
        user = AnonymousUser()
        user.email = 'nobody@plugit-standalone.ebuio'
        user.first_name = 'Ano'
        user.last_name = 'Nymous'
        user.ebuio_member = False
        user.ebuio_admin = False
        user.subscription_labels = []
    elif settings.PIAPI_STANDALONE and pk >= 0:
        # Generate an unknown user for compatibility reason in standalone mode
        user = DUser(pk=pk, username='Logged', first_name='Unknown', last_name='Other User', email='unknown@plugit-standalone.ebuio')
        user.gravatar = 'https://www.gravatar.com/avatar/unknown?d=retro'
        user.ebuio_member = False
        user.ebuio_admin = False
        user.subscription_labels = []

    if user:
        user.ebuio_orga_member = user.ebuio_member
        user.ebuio_orga_admin = user.ebuio_admin

    return user