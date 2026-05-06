def setup_auth_policies(config, raml_root):
    """ Setup authentication, authorization policies.

    Performs basic validation to check all the required values are present
    and performs authentication, authorization policies generation using
    generator functions from `AUTHENTICATION_POLICIES`.

    :param config: Pyramid Configurator instance.
    :param raml_root: Instance of ramlfications.raml.RootNode.
    """
    log.info('Configuring auth policies')
    secured_by_all = raml_root.secured_by or []
    secured_by = [item for item in secured_by_all if item]
    if not secured_by:
        log.info('API is not secured. `secured_by` attribute '
                 'value missing.')
        return
    secured_by = secured_by[0]

    schemes = {scheme.name: scheme
               for scheme in raml_root.security_schemes}
    if secured_by not in schemes:
        raise ValueError(
            'Undefined security scheme used in `secured_by`: {}'.format(
                secured_by))

    scheme = schemes[secured_by]
    if scheme.type not in AUTHENTICATION_POLICIES:
        raise ValueError('Unsupported security scheme type: {}'.format(
            scheme.type))

    # Setup Authentication policy
    policy_generator = AUTHENTICATION_POLICIES[scheme.type]
    params = dictset(scheme.settings or {})
    authn_policy = policy_generator(config, params)
    config.set_authentication_policy(authn_policy)

    # Setup Authorization policy
    authz_policy = ACLAuthorizationPolicy()
    config.set_authorization_policy(authz_policy)