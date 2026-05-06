def harvest_fundref(source=None):
    """Harvest funders from FundRef and store as authority records."""
    loader = LocalFundRefLoader(source=source) if source \
        else RemoteFundRefLoader()
    for funder_json in loader.iter_funders():
        register_funder.delay(funder_json)