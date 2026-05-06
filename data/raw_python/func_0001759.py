def get_soup(url, downloader=None, user_agent=None, user_agent_config_yaml=None, user_agent_lookup=None, **kwargs):
    # type: (str, Download, Optional[str], Optional[str], Optional[str], Any) -> BeautifulSoup
    """
    Get BeautifulSoup object for a url. Requires either global user agent to be set or appropriate user agent
    parameter(s) to be completed.

    Args:
        url (str): url to read
        downloader (Download): Download object. Defaults to creating a Download object with given user agent values.
        user_agent (Optional[str]): User agent string. HDXPythonUtilities/X.X.X- is prefixed.
        user_agent_config_yaml (Optional[str]): Path to YAML user agent configuration. Ignored if user_agent supplied. Defaults to ~/.useragent.yml.
        user_agent_lookup (Optional[str]): Lookup key for YAML. Ignored if user_agent supplied.

    Returns:
        BeautifulSoup: The BeautifulSoup object for a url

    """
    if not downloader:
        downloader = Download(user_agent, user_agent_config_yaml, user_agent_lookup, **kwargs)
    response = downloader.download(url)
    return BeautifulSoup(response.text, 'html.parser')