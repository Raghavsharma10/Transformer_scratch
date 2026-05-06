def get_gcloud_pricelist():
    """Retrieve latest pricelist from Google Cloud, or use
    cached copy if not reachable.
    """
    try:
        r = requests.get('http://cloudpricingcalculator.appspot.com'
                         '/static/data/pricelist.json')
        content = json.loads(r.content)
    except ConnectionError:
        logger.warning(
            "Couldn't get updated pricelist from "
            "http://cloudpricingcalculator.appspot.com"
            "/static/data/pricelist.json. Falling back to cached "
            "copy, but prices may be out of date.")
        with open('gcloudpricelist.json') as infile:
            content = json.load(infile)

    pricelist = content['gcp_price_list']
    return pricelist