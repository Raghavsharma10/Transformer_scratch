def fetch_csv_dataframe(
        download_url,
        filename=None,
        subdir=None,
        **pandas_kwargs):
    """
    Download a remote file from `download_url` and save it locally as `filename`.
    Load that local file as a CSV into Pandas using extra keyword arguments such as sep='\t'.
    """
    path = fetch_file(
        download_url=download_url,
        filename=filename,
        decompress=True,
        subdir=subdir)
    return pd.read_csv(path, **pandas_kwargs)