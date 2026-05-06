def entity(ctx, debug, uncolorize, **kwargs):
    """
    CLI for tonomi.com using contrib-python-qubell-client

    To enable completion:

      eval "$(_NOMI_COMPLETE=source nomi)"
    """
    global PROVIDER_CONFIG

    if debug:
        log.basicConfig(level=log.DEBUG)
        log.getLogger("requests.packages.urllib3.connectionpool").setLevel(log.DEBUG)
    for (k, v) in kwargs.iteritems():
        if v:
            QUBELL[k] = v
    PROVIDER_CONFIG = {
        'configuration.provider': PROVIDER['provider_type'],
        'configuration.legacy-regions': PROVIDER['provider_region'],
        'configuration.endpoint-url': '',
        'configuration.legacy-security-group': '',
        'configuration.identity': PROVIDER['provider_identity'],
        'configuration.credential': PROVIDER['provider_credential']
    }

    class UserContext(object):
        def __init__(self):
            self.platform = None
            self.unauthenticated_platform = None
            self.colorize = not (uncolorize)

        def get_platform(self):
            if not self.platform:
                assert QUBELL["tenant"], "No platform URL provided. Set QUBELL_TENANT or use --tenant option."
                if not QUBELL["token"]:
                    assert QUBELL["user"], "No username. Set QUBELL_USER or use --user option."
                    assert QUBELL["password"], "No password provided. Set QUBELL_PASSWORD or use --password option."

                self.platform = QubellPlatform.connect(
                    tenant=QUBELL["tenant"],
                    user=QUBELL["user"],
                    password=QUBELL["password"],
                    token=QUBELL["token"])
            return self.platform

        def get_unauthenticated_platform(self):
            if not self.unauthenticated_platform:
                assert QUBELL["tenant"], "No platform URL provided. Set QUBELL_TENANT or use --tenant option."

                self.unauthenticated_platform = QubellPlatform.connect(tenant=QUBELL["tenant"])

            return self.unauthenticated_platform

    ctx = click.get_current_context()
    ctx.obj = UserContext()