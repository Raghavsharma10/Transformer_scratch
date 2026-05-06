def loadgrants(source=None, setspec=None, all_grants=False):
    """Harvest grants from OpenAIRE.

    :param source: Load the grants from a local sqlite db (offline).
        The value of the parameter should be a path to the local file.
    :type source: str
    :param setspec: Harvest specific set through OAI-PMH
        Creates a remote connection to OpenAIRE.
    :type setspec: str
    :param all_grants: Harvest all sets through OAI-PMH,
        as specified in the configuration OPENAIRE_GRANTS_SPEC. Sets are
        harvested sequentially in the order specified in the configuration.
        Creates a remote connection to OpenAIRE.
    :type all_grants: bool
    """
    assert all_grants or setspec or source, \
        "Either '--all', '--setspec' or '--source' is required parameter."
    if all_grants:
        harvest_all_openaire_projects.delay()
    elif setspec:
        click.echo("Remote grants loading sent to queue.")
        harvest_openaire_projects.delay(setspec=setspec)
    else:  # if source
        loader = LocalOAIRELoader(source=source)
        loader._connect()
        cnt = loader._count()
        click.echo("Sending grants to queue.")
        with click.progressbar(loader.iter_grants(), length=cnt) as grants_bar:

            for grant_json in grants_bar:
                register_grant.delay(grant_json)