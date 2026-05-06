def resumption_token(parent, pagination, **kwargs):
    """Attach resumption token element to a parent."""
    # Do not add resumptionToken if all results fit to the first page.
    if pagination.page == 1 and not pagination.has_next:
        return

    token = serialize(pagination, **kwargs)
    e_resumptionToken = SubElement(parent, etree.QName(NS_OAIPMH,
                                                       'resumptionToken'))
    if pagination.total:
        expiration_date = datetime.utcnow() + timedelta(
            seconds=current_app.config[
                'OAISERVER_RESUMPTION_TOKEN_EXPIRE_TIME'
            ]
        )
        e_resumptionToken.set('expirationDate', datetime_to_datestamp(
            expiration_date
        ))
        e_resumptionToken.set('cursor', str(
            (pagination.page - 1) * pagination.per_page
        ))
        e_resumptionToken.set('completeListSize', str(pagination.total))

    if token:
        e_resumptionToken.text = token