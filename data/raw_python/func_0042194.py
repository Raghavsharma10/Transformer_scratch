def _make_rofr_rdf(app, api_home_dir, api_uri):
    """
    The setup function that creates the Register of Registers.

    Do not call from outside setup
    :param app: the Flask app containing this LDAPI
    :type app: Flask app
    :param api_uri: URI base of the API
    :type api_uri: string
    :return: none
    :rtype: None
    """
    from time import sleep
    from pyldapi import RegisterRenderer, RegisterOfRegistersRenderer
    try:
        os.remove(os.path.join(api_home_dir, 'rofr.ttl'))
    except FileNotFoundError:
        pass
    sleep(1)  # to ensure that this occurs after the Flask boot
    print('making RofR')
    g = Graph()
    # get the RDF for each Register, extract the bits we need, write them to graph g
    for rule in app.url_map.iter_rules():
        if '<' not in str(rule):  # no registers can have a Flask variable in their path
            # make the register view URI for each possible register
            try:
                endpoint_func = app.view_functions[rule.endpoint]
            except (AttributeError, KeyError):
                continue
            try:
                candidate_register_uri = api_uri + str(
                    rule) + '?_view=reg&_format=_internal'
                test_context = app.test_request_context(candidate_register_uri)
                with test_context:
                    resp = endpoint_func()
            except RegOfRegTtlError:  # usually an RofR renderer cannot find its rofr.ttl.
                continue
            except Exception as e:
                raise e
            if isinstance(resp, RegisterOfRegistersRenderer):
                continue  # forbid adding a register of registers to a register of registers.
            if isinstance(resp, RegisterRenderer):
                with test_context:
                    try:
                        resp.format = 'text/html'
                        html_resp = resp._render_reg_view_html()
                    except TemplateNotFound:  # missing html template
                        pass  # TODO: Fail on this error
                    resp.format = 'application/json'
                    json_resp = resp._render_reg_view_json()
                    resp.format = 'text/turtle'
                    rdf_resp = resp._render_reg_view_rdf()

                _filter_register_graph(
                    candidate_register_uri.replace('?_view=reg&_format=_internal', ''),
                    rdf_resp, g)

    # serialise g
    with open(os.path.join(api_home_dir, 'rofr.ttl'), 'w') as f:
        f.write(g.serialize(format='text/turtle').decode('utf-8'))

    print('finished making RofR')