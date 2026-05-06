def get_shipping_label(jobIds=None, name=None, company=None, phoneNumber=None, country=None, stateOrProvince=None, city=None, postalCode=None, street1=None, street2=None, street3=None, APIVersion=None):
    """
    This operation generates a pre-paid UPS shipping label that you will use to ship your device to AWS for processing.
    See also: AWS API Documentation
    
    
    :example: response = client.get_shipping_label(
        jobIds=[
            'string',
        ],
        name='string',
        company='string',
        phoneNumber='string',
        country='string',
        stateOrProvince='string',
        city='string',
        postalCode='string',
        street1='string',
        street2='string',
        street3='string',
        APIVersion='string'
    )
    
    
    :type jobIds: list
    :param jobIds: [REQUIRED]
            (string) --
            

    :type name: string
    :param name: Specifies the name of the person responsible for shipping this package.

    :type company: string
    :param company: Specifies the name of the company that will ship this package.

    :type phoneNumber: string
    :param phoneNumber: Specifies the phone number of the person responsible for shipping this package.

    :type country: string
    :param country: Specifies the name of your country for the return address.

    :type stateOrProvince: string
    :param stateOrProvince: Specifies the name of your state or your province for the return address.

    :type city: string
    :param city: Specifies the name of your city for the return address.

    :type postalCode: string
    :param postalCode: Specifies the postal code for the return address.

    :type street1: string
    :param street1: Specifies the first part of the street address for the return address, for example 1234 Main Street.

    :type street2: string
    :param street2: Specifies the optional second part of the street address for the return address, for example Suite 100.

    :type street3: string
    :param street3: Specifies the optional third part of the street address for the return address, for example c/o Jane Doe.

    :type APIVersion: string
    :param APIVersion: Specifies the version of the client tool.

    :rtype: dict
    :return: {
        'ShippingLabelURL': 'string',
        'Warning': 'string'
    }
    
    
    :returns: 
    (dict) --
    ShippingLabelURL (string) --
    Warning (string) --
    
    
    
    """
    pass