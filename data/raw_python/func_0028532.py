def get_template_details(template_name, auth, url):
    """Uses the get_template_id() funct to gather the template_id to craft a
    get_template_details_url which is sent to the IMC server using
    a get Method
    :param template_name: str containing the entire contents of the configuration segment

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: If successful, return dict containing the template details

    :rtype: dict

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.icc import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> filecontent = 'sample file content'

    >>> create_new_file = create_cfg_segment('CW7SNMP.cfg',
                                              filecontent,
                                              'My New Template',
                                               auth.creds,
                                               auth.url)

    >>> template_contents = get_template_details('CW7SNMP.cfg', auth.creds, auth.url)

    >>> assert type(template_contents) is dict

    """
    file_id = get_template_id(template_name, auth, url)
    if isinstance(file_id, str):
        return file_id
    f_url = url + "/imcrs/icc/confFile/" + str(file_id)
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            template_details = json.loads(response.text)
            return template_details
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_template_contents: An Error has occured"