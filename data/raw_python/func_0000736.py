def _setup_ticket_policy(config, params):
    """ Setup Pyramid AuthTktAuthenticationPolicy.

    Notes:
      * Initial `secret` params value is considered to be a name of config
        param that represents a cookie name.
      * `auth_model.get_groups_by_userid` is used as a `callback`.
      * Also connects basic routes to perform authentication actions.

    :param config: Pyramid Configurator instance.
    :param params: Nefertari dictset which contains security scheme
        `settings`.
    """
    from nefertari.authentication.views import (
        TicketAuthRegisterView, TicketAuthLoginView,
        TicketAuthLogoutView)

    log.info('Configuring Pyramid Ticket Authn policy')
    if 'secret' not in params:
        raise ValueError(
            'Missing required security scheme settings: secret')
    params['secret'] = config.registry.settings[params['secret']]

    auth_model = config.registry.auth_model
    params['callback'] = auth_model.get_groups_by_userid

    config.add_request_method(
        auth_model.get_authuser_by_userid, 'user', reify=True)

    policy = AuthTktAuthenticationPolicy(**params)

    RegisterViewBase = TicketAuthRegisterView
    if config.registry.database_acls:
        class RegisterViewBase(ACLAssignRegisterMixin,
                               TicketAuthRegisterView):
            pass

    class RamsesTicketAuthRegisterView(RegisterViewBase):
        Model = config.registry.auth_model

    class RamsesTicketAuthLoginView(TicketAuthLoginView):
        Model = config.registry.auth_model

    class RamsesTicketAuthLogoutView(TicketAuthLogoutView):
        Model = config.registry.auth_model

    common_kw = {
        'prefix': 'auth',
        'factory': 'nefertari.acl.AuthenticationACL',
    }

    root = config.get_root_resource()
    root.add('register', view=RamsesTicketAuthRegisterView, **common_kw)
    root.add('login', view=RamsesTicketAuthLoginView, **common_kw)
    root.add('logout', view=RamsesTicketAuthLogoutView, **common_kw)

    return policy