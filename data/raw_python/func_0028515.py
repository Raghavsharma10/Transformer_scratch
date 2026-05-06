def create_cfg_segment(filename, filecontent, description, auth, url):
    '''
    Takes a str into var filecontent which represents the entire content of a configuration segment, or partial
    configuration file. Takes a str into var description which represents the description of the configuration segment
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

    >>> filecontent = ("""sample file content""")
     >>> create_new_file = create_cfg_segment('CW7SNMP.cfg', filecontent, 'My New Template', auth.creds, auth.url)

     >>> template_id = get_template_id('CW7SNMP.cfg', auth.creds, auth.url)

     >>> assert type(template_id) is str

     >>>
    '''
    payload = {"confFileName": filename,
               "confFileType": "2",
               "cfgFileParent": "-1",
               "confFileDesc": description,
               "content": filecontent}
    create_cfg_segment_url = "/imcrs/icc/confFile"
    f_url = url + create_cfg_segment_url
    # creates the URL using the payload variable as the contents
    r = requests.post(f_url,data= (json.dumps(payload)), auth=auth, headers=HEADERS)
    try:
        if r.status_code == 201:
            return True
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " create_cfg_segment: An Error has occured"