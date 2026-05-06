def run_airbnb_demo(data_dir):
    """HyperTransfomer will transform back and forth data airbnb data."""

    # Setup
    meta_file = os.path.join(data_dir, 'Airbnb_demo_meta.json')
    transformer_list = ['NumberTransformer', 'DTTransformer', 'CatTransformer']
    ht = HyperTransformer(meta_file)

    # Run
    transformed = ht.fit_transform(transformer_list=transformer_list)
    result = ht.reverse_transform(tables=transformed)

    # Check
    assert result.keys() == ht.table_dict.keys()

    for name, table in result.items():
        assert not result[name].isnull().all().all()