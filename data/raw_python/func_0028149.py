def create_remoteckan(cls, site_url, user_agent=None, user_agent_config_yaml=None, user_agent_lookup=None,
                          session=None, **kwargs):
        # type: (str, Optional[str], Optional[str], Optional[str], requests.Session, Any) -> ckanapi.RemoteCKAN
        """
        Create remote CKAN instance from configuration

        Args:
            site_url (str): Site url.
            user_agent (Optional[str]): User agent string. HDXPythonLibrary/X.X.X- is prefixed.
            user_agent_config_yaml (Optional[str]): Path to YAML user agent configuration. Ignored if user_agent supplied. Defaults to ~/.useragent.yml.
            user_agent_lookup (Optional[str]): Lookup key for YAML. Ignored if user_agent supplied.
            session (requests.Session): requests Session object to use. Defaults to calling hdx.utilities.session.get_session()

        Returns:
            ckanapi.RemoteCKAN: Remote CKAN instance

        """
        if not session:
            session = get_session(user_agent, user_agent_config_yaml, user_agent_lookup, prefix=Configuration.prefix,
                                  method_whitelist=frozenset(['HEAD', 'TRACE', 'GET', 'POST', 'PUT',
                                                              'OPTIONS', 'DELETE']), **kwargs)
            ua = session.headers['User-Agent']
        else:
            ua = kwargs.get('full_agent')
            if not ua:
                ua = UserAgent.get(user_agent, user_agent_config_yaml, user_agent_lookup, prefix=Configuration.prefix,
                                   **kwargs)
        return ckanapi.RemoteCKAN(site_url, user_agent=ua, session=session)