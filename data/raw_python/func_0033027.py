def jsonresolver_loader(url_map):
    """Jsonresolver hook for funders resolving."""
    def endpoint(doi_code):
        pid_value = "10.13039/{0}".format(doi_code)
        _, record = Resolver(pid_type='frdoi', object_type='rec',
                             getter=Record.get_record).resolve(pid_value)
        return record

    pattern = '/10.13039/<doi_code>'
    url_map.add(Rule(pattern, endpoint=endpoint, host='doi.org'))
    url_map.add(Rule(pattern, endpoint=endpoint, host='dx.doi.org'))