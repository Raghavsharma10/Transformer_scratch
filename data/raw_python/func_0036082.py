def get(dataset = None, include_metadata = False, mnemonics = None, **dim_values):
    """Use this function to get data from Knoema dataset."""

    if not dataset and not mnemonics:
        raise ValueError('Dataset id is not specified')

    if mnemonics and dim_values:
        raise ValueError('The function does not support specifying mnemonics and selection in a single call')

    config = ApiConfig()
    client = ApiClient(config.host, config.app_id, config.app_secret)
    client.check_correct_host()

    ds = client.get_dataset(dataset) if dataset else None
    reader =  MnemonicsDataReader(client, mnemonics) if mnemonics else StreamingDataReader(client, dim_values) if ds.type == 'Regular' else PivotDataReader(client, dim_values)
    reader.include_metadata = include_metadata
    reader.dataset = ds

    return reader.get_pandasframe()