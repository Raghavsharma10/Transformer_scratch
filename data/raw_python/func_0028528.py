def create_cfg_segment(filename, filecontent, description, auth, url):
    """
    Takes a str into var filecontent which represents the entire content of a configuration
    segment, or partial configuration file. Takes a str into var description which represents the
    description of the configuration segment
    :param filename: str containing the name of the configuration segment.

    :param filecontent: str containing the entire contents of the configuration segment

    :param description: str contrianing the description of the configuration segment

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: If successful, Boolena of type True

    :rtype: Boolean

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.icc import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> filecontent = 'sample file content'

    >>> create_new_file = create_cfg_segment('CW7SNMP.cfg',
                                              filecontent,
                                              'My New Template',
                                               auth.creds,
                                               auth.url)

    >>> template_id = get_template_id('CW7SNMP.cfg', auth.creds, auth.url)

    >>> assert type(template_id) is str

    >>>
    """
    payload = {"confFileName": filename,
               "confFileType": "2",
               "cfgFileParent": "-1",
               "confFileDesc": description,
               "content": filecontent}
    f_url = url + "/imcrs/icc/confFile"
    response = requests.post(f_url, data=(json.dumps(payload)), auth=auth, headers=HEADERS)
    try:
        if response.status_code == 201:
            print("Template successfully created")
            return response.status_code
        elif response.status_code is not 201:
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " create_cfg_segment: An Error has occured"