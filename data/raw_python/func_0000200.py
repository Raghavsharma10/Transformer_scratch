def get_manifest_from_meta(metaurl, name):
    """
    Extact manifest url from metadata url
    :param metaurl: Url to metadata
    :param name: Name of application to extract
    :return:
    """
    if 'http' in metaurl:
        kit = yaml.safe_load(requests.get(url=metaurl).content)['kit']['applications']
    else:
        kit = yaml.safe_load(open(metaurl).read())['kit']['applications']
    app_urls = [x['manifest'] for x in kit if x['name'] == name]
    assert len(app_urls) == 1
    return app_urls[0]