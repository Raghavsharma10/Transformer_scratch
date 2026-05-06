def get_starter_kit_meta(name):
    """
    Extract metadata link for starter kit from platform configs. Starter kit available on add component - starter kit menu.
    Beware, config could be changed by deploy scripts during deploy.
    :param name: Name of starter kit
    :return: Link to metadata
    """
    kits = yaml.safe_load(requests.get(url=starter_kits_url).content)['kits']
    kits_meta_url = [x['metaUrl'] for x in kits if x['name'] == name]

    assert len(kits_meta_url)==1, "No component %s found in meta:\n %s" % (name, kits)
    meta = yaml.safe_load(requests.get(url=kits_meta_url[0]).content)['download_url']
    return meta