def harvest_openaire_projects(source=None, setspec=None):
    """Harvest grants from OpenAIRE and store as authority records."""
    loader = LocalOAIRELoader(source=source) if source \
        else RemoteOAIRELoader(setspec=setspec)
    for grant_json in loader.iter_grants():
        register_grant.delay(grant_json)