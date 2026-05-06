def setup(provider=None):
    """
    Creates the provider config files needed to deploy your project
    """
    site = init(provider)
    if not site:
        site = yaml.safe_load(_read_file(DEPLOY_YAML))

    provider_class = PROVIDERS[site['provider']]
    provider_class.init(site)