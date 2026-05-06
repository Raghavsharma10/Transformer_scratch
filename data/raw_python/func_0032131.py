def resolve_grant_endpoint(doi_grant_code):
    """Resolve the OpenAIRE grant."""
    # jsonresolver will evaluate current_app on import if outside of function.
    from flask import current_app
    pid_value = '10.13039/{0}'.format(doi_grant_code)
    try:
        _, record = Resolver(pid_type='grant', object_type='rec',
                             getter=Record.get_record).resolve(pid_value)
        return record
    except Exception:
        current_app.logger.error(
            'Grant {0} does not exists.'.format(pid_value), exc_info=True)
        raise