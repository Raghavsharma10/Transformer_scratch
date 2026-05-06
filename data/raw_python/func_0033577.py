def deploy(provider=None):
    """
    Deploys your project
    """
    if os.path.exists(DEPLOY_YAML):
        site = yaml.safe_load(_read_file(DEPLOY_YAML))

    provider_class = PROVIDERS[site['provider']]
    provider_class.deploy()