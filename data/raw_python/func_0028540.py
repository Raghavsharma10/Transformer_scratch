def get_plat_operator(auth, url):
    """
    Funtion takes no inputs and returns a list of dictionaties of all of the operators currently
    configured on the HPE IMC system

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: list of dictionaries where each element represents one operator

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.operator import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> plat_operators = get_plat_operator(auth.creds, auth.url)

    >>> assert type(plat_operators) is list

    >>> assert 'name' in plat_operators[0]

    """
    f_url = url + '/imcrs/plat/operator?start=0&size=1000&orderBy=id&desc=false&total=false'
    try:
        response = requests.get(f_url, auth=auth, headers=HEADERS)
        plat_oper_list = json.loads(response.text)['operator']
        if isinstance(plat_oper_list, dict):
            oper_list = [plat_oper_list]
            return oper_list
        return plat_oper_list
    except requests.exceptions.RequestException as error:
        print("Error:\n" + str(error) + ' get_plat_operator: An Error has occured')
        return "Error:\n" + str(error) + ' get_plat_operator: An Error has occured'