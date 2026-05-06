def get_dev_asset_details_all(auth, url):
    """Takes no input to fetch device assett details from HP IMC RESTFUL API

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: list of dictionatires containing the device asset details

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.netassets import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> all_assets = get_dev_asset_details_all( auth.creds, auth.url)

    >>> assert type(all_assets) is list

    >>> assert 'asset' in all_assets[0]

    """
    get_dev_asset_details_all_url = "/imcrs/netasset/asset?start=0&size=15000"
    f_url = url + get_dev_asset_details_all_url
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=HEADERS)
    # r.status_code
    try:
        if r.status_code == 200:
            dev_asset_info = (json.loads(r.text))['netAsset']
            return dev_asset_info
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + ' get_dev_asset_details: An Error has occured'