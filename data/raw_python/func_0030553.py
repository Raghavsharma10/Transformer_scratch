def get_repo_url(mead_tag, nexus_base_url, prefix="hudson-", suffix=""):
    """
    Creates repository Nexus group URL composed of:
        <nexus_base_url>/content/groups/<prefix><mead_tag><suffix>

    :param mead_tag: name of the MEAD tag used to create the proxy URL in settings.xml
    :param nexus_base_url: the base URL of a Nexus instance
    :param prefix: Nexus group name prefix, default is "hudson-"
    :param suffix: Nexus group name suffix, e.g. "-jboss-central" or "-reverse"
    :returns:
    """
    result = urlparse.urljoin(nexus_base_url, "content/groups/")
    result = urlparse.urljoin(result, "%s%s%s/" % (prefix, mead_tag, suffix))
    return result