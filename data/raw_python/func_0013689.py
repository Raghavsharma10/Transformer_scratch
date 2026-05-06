def transfer_domain(DomainName=None, IdnLangCode=None, DurationInYears=None, Nameservers=None, AuthCode=None, AutoRenew=None, AdminContact=None, RegistrantContact=None, TechContact=None, PrivacyProtectAdminContact=None, PrivacyProtectRegistrantContact=None, PrivacyProtectTechContact=None):
    """
    This operation transfers a domain from another registrar to Amazon Route 53. When the transfer is complete, the domain is registered with the AWS registrar partner, Gandi.
    For transfer requirements, a detailed procedure, and information about viewing the status of a domain transfer, see Transferring Registration for a Domain to Amazon Route 53 in the Amazon Route 53 Developer Guide .
    If the registrar for your domain is also the DNS service provider for the domain, we highly recommend that you consider transferring your DNS service to Amazon Route 53 or to another DNS service provider before you transfer your registration. Some registrars provide free DNS service when you purchase a domain registration. When you transfer the registration, the previous registrar will not renew your domain registration and could end your DNS service at any time.
    If the transfer is successful, this method returns an operation ID that you can use to track the progress and completion of the action. If the transfer doesn't complete successfully, the domain registrant will be notified by email.
    See also: AWS API Documentation
    
    
    :example: response = client.transfer_domain(
        DomainName='string',
        IdnLangCode='string',
        DurationInYears=123,
        Nameservers=[
            {
                'Name': 'string',
                'GlueIps': [
                    'string',
                ]
            },
        ],
        AuthCode='string',
        AutoRenew=True|False,
        AdminContact={
            'FirstName': 'string',
            'LastName': 'string',
            'ContactType': 'PERSON'|'COMPANY'|'ASSOCIATION'|'PUBLIC_BODY'|'RESELLER',
            'OrganizationName': 'string',
            'AddressLine1': 'string',
            'AddressLine2': 'string',
            'City': 'string',
            'State': 'string',
            'CountryCode': 'AD'|'AE'|'AF'|'AG'|'AI'|'AL'|'AM'|'AN'|'AO'|'AQ'|'AR'|'AS'|'AT'|'AU'|'AW'|'AZ'|'BA'|'BB'|'BD'|'BE'|'BF'|'BG'|'BH'|'BI'|'BJ'|'BL'|'BM'|'BN'|'BO'|'BR'|'BS'|'BT'|'BW'|'BY'|'BZ'|'CA'|'CC'|'CD'|'CF'|'CG'|'CH'|'CI'|'CK'|'CL'|'CM'|'CN'|'CO'|'CR'|'CU'|'CV'|'CX'|'CY'|'CZ'|'DE'|'DJ'|'DK'|'DM'|'DO'|'DZ'|'EC'|'EE'|'EG'|'ER'|'ES'|'ET'|'FI'|'FJ'|'FK'|'FM'|'FO'|'FR'|'GA'|'GB'|'GD'|'GE'|'GH'|'GI'|'GL'|'GM'|'GN'|'GQ'|'GR'|'GT'|'GU'|'GW'|'GY'|'HK'|'HN'|'HR'|'HT'|'HU'|'ID'|'IE'|'IL'|'IM'|'IN'|'IQ'|'IR'|'IS'|'IT'|'JM'|'JO'|'JP'|'KE'|'KG'|'KH'|'KI'|'KM'|'KN'|'KP'|'KR'|'KW'|'KY'|'KZ'|'LA'|'LB'|'LC'|'LI'|'LK'|'LR'|'LS'|'LT'|'LU'|'LV'|'LY'|'MA'|'MC'|'MD'|'ME'|'MF'|'MG'|'MH'|'MK'|'ML'|'MM'|'MN'|'MO'|'MP'|'MR'|'MS'|'MT'|'MU'|'MV'|'MW'|'MX'|'MY'|'MZ'|'NA'|'NC'|'NE'|'NG'|'NI'|'NL'|'NO'|'NP'|'NR'|'NU'|'NZ'|'OM'|'PA'|'PE'|'PF'|'PG'|'PH'|'PK'|'PL'|'PM'|'PN'|'PR'|'PT'|'PW'|'PY'|'QA'|'RO'|'RS'|'RU'|'RW'|'SA'|'SB'|'SC'|'SD'|'SE'|'SG'|'SH'|'SI'|'SK'|'SL'|'SM'|'SN'|'SO'|'SR'|'ST'|'SV'|'SY'|'SZ'|'TC'|'TD'|'TG'|'TH'|'TJ'|'TK'|'TL'|'TM'|'TN'|'TO'|'TR'|'TT'|'TV'|'TW'|'TZ'|'UA'|'UG'|'US'|'UY'|'UZ'|'VA'|'VC'|'VE'|'VG'|'VI'|'VN'|'VU'|'WF'|'WS'|'YE'|'YT'|'ZA'|'ZM'|'ZW',
            'ZipCode': 'string',
            'PhoneNumber': 'string',
            'Email': 'string',
            'Fax': 'string',
            'ExtraParams': [
                {
                    'Name': 'DUNS_NUMBER'|'BRAND_NUMBER'|'BIRTH_DEPARTMENT'|'BIRTH_DATE_IN_YYYY_MM_DD'|'BIRTH_COUNTRY'|'BIRTH_CITY'|'DOCUMENT_NUMBER'|'AU_ID_NUMBER'|'AU_ID_TYPE'|'CA_LEGAL_TYPE'|'CA_BUSINESS_ENTITY_TYPE'|'ES_IDENTIFICATION'|'ES_IDENTIFICATION_TYPE'|'ES_LEGAL_FORM'|'FI_BUSINESS_NUMBER'|'FI_ID_NUMBER'|'IT_PIN'|'RU_PASSPORT_DATA'|'SE_ID_NUMBER'|'SG_ID_NUMBER'|'VAT_NUMBER',
                    'Value': 'string'
                },
            ]
        },
        RegistrantContact={
            'FirstName': 'string',
            'LastName': 'string',
            'ContactType': 'PERSON'|'COMPANY'|'ASSOCIATION'|'PUBLIC_BODY'|'RESELLER',
            'OrganizationName': 'string',
            'AddressLine1': 'string',
            'AddressLine2': 'string',
            'City': 'string',
            'State': 'string',
            'CountryCode': 'AD'|'AE'|'AF'|'AG'|'AI'|'AL'|'AM'|'AN'|'AO'|'AQ'|'AR'|'AS'|'AT'|'AU'|'AW'|'AZ'|'BA'|'BB'|'BD'|'BE'|'BF'|'BG'|'BH'|'BI'|'BJ'|'BL'|'BM'|'BN'|'BO'|'BR'|'BS'|'BT'|'BW'|'BY'|'BZ'|'CA'|'CC'|'CD'|'CF'|'CG'|'CH'|'CI'|'CK'|'CL'|'CM'|'CN'|'CO'|'CR'|'CU'|'CV'|'CX'|'CY'|'CZ'|'DE'|'DJ'|'DK'|'DM'|'DO'|'DZ'|'EC'|'EE'|'EG'|'ER'|'ES'|'ET'|'FI'|'FJ'|'FK'|'FM'|'FO'|'FR'|'GA'|'GB'|'GD'|'GE'|'GH'|'GI'|'GL'|'GM'|'GN'|'GQ'|'GR'|'GT'|'GU'|'GW'|'GY'|'HK'|'HN'|'HR'|'HT'|'HU'|'ID'|'IE'|'IL'|'IM'|'IN'|'IQ'|'IR'|'IS'|'IT'|'JM'|'JO'|'JP'|'KE'|'KG'|'KH'|'KI'|'KM'|'KN'|'KP'|'KR'|'KW'|'KY'|'KZ'|'LA'|'LB'|'LC'|'LI'|'LK'|'LR'|'LS'|'LT'|'LU'|'LV'|'LY'|'MA'|'MC'|'MD'|'ME'|'MF'|'MG'|'MH'|'MK'|'ML'|'MM'|'MN'|'MO'|'MP'|'MR'|'MS'|'MT'|'MU'|'MV'|'MW'|'MX'|'MY'|'MZ'|'NA'|'NC'|'NE'|'NG'|'NI'|'NL'|'NO'|'NP'|'NR'|'NU'|'NZ'|'OM'|'PA'|'PE'|'PF'|'PG'|'PH'|'PK'|'PL'|'PM'|'PN'|'PR'|'PT'|'PW'|'PY'|'QA'|'RO'|'RS'|'RU'|'RW'|'SA'|'SB'|'SC'|'SD'|'SE'|'SG'|'SH'|'SI'|'SK'|'SL'|'SM'|'SN'|'SO'|'SR'|'ST'|'SV'|'SY'|'SZ'|'TC'|'TD'|'TG'|'TH'|'TJ'|'TK'|'TL'|'TM'|'TN'|'TO'|'TR'|'TT'|'TV'|'TW'|'TZ'|'UA'|'UG'|'US'|'UY'|'UZ'|'VA'|'VC'|'VE'|'VG'|'VI'|'VN'|'VU'|'WF'|'WS'|'YE'|'YT'|'ZA'|'ZM'|'ZW',
            'ZipCode': 'string',
            'PhoneNumber': 'string',
            'Email': 'string',
            'Fax': 'string',
            'ExtraParams': [
                {
                    'Name': 'DUNS_NUMBER'|'BRAND_NUMBER'|'BIRTH_DEPARTMENT'|'BIRTH_DATE_IN_YYYY_MM_DD'|'BIRTH_COUNTRY'|'BIRTH_CITY'|'DOCUMENT_NUMBER'|'AU_ID_NUMBER'|'AU_ID_TYPE'|'CA_LEGAL_TYPE'|'CA_BUSINESS_ENTITY_TYPE'|'ES_IDENTIFICATION'|'ES_IDENTIFICATION_TYPE'|'ES_LEGAL_FORM'|'FI_BUSINESS_NUMBER'|'FI_ID_NUMBER'|'IT_PIN'|'RU_PASSPORT_DATA'|'SE_ID_NUMBER'|'SG_ID_NUMBER'|'VAT_NUMBER',
                    'Value': 'string'
                },
            ]
        },
        TechContact={
            'FirstName': 'string',
            'LastName': 'string',
            'ContactType': 'PERSON'|'COMPANY'|'ASSOCIATION'|'PUBLIC_BODY'|'RESELLER',
            'OrganizationName': 'string',
            'AddressLine1': 'string',
            'AddressLine2': 'string',
            'City': 'string',
            'State': 'string',
            'CountryCode': 'AD'|'AE'|'AF'|'AG'|'AI'|'AL'|'AM'|'AN'|'AO'|'AQ'|'AR'|'AS'|'AT'|'AU'|'AW'|'AZ'|'BA'|'BB'|'BD'|'BE'|'BF'|'BG'|'BH'|'BI'|'BJ'|'BL'|'BM'|'BN'|'BO'|'BR'|'BS'|'BT'|'BW'|'BY'|'BZ'|'CA'|'CC'|'CD'|'CF'|'CG'|'CH'|'CI'|'CK'|'CL'|'CM'|'CN'|'CO'|'CR'|'CU'|'CV'|'CX'|'CY'|'CZ'|'DE'|'DJ'|'DK'|'DM'|'DO'|'DZ'|'EC'|'EE'|'EG'|'ER'|'ES'|'ET'|'FI'|'FJ'|'FK'|'FM'|'FO'|'FR'|'GA'|'GB'|'GD'|'GE'|'GH'|'GI'|'GL'|'GM'|'GN'|'GQ'|'GR'|'GT'|'GU'|'GW'|'GY'|'HK'|'HN'|'HR'|'HT'|'HU'|'ID'|'IE'|'IL'|'IM'|'IN'|'IQ'|'IR'|'IS'|'IT'|'JM'|'JO'|'JP'|'KE'|'KG'|'KH'|'KI'|'KM'|'KN'|'KP'|'KR'|'KW'|'KY'|'KZ'|'LA'|'LB'|'LC'|'LI'|'LK'|'LR'|'LS'|'LT'|'LU'|'LV'|'LY'|'MA'|'MC'|'MD'|'ME'|'MF'|'MG'|'MH'|'MK'|'ML'|'MM'|'MN'|'MO'|'MP'|'MR'|'MS'|'MT'|'MU'|'MV'|'MW'|'MX'|'MY'|'MZ'|'NA'|'NC'|'NE'|'NG'|'NI'|'NL'|'NO'|'NP'|'NR'|'NU'|'NZ'|'OM'|'PA'|'PE'|'PF'|'PG'|'PH'|'PK'|'PL'|'PM'|'PN'|'PR'|'PT'|'PW'|'PY'|'QA'|'RO'|'RS'|'RU'|'RW'|'SA'|'SB'|'SC'|'SD'|'SE'|'SG'|'SH'|'SI'|'SK'|'SL'|'SM'|'SN'|'SO'|'SR'|'ST'|'SV'|'SY'|'SZ'|'TC'|'TD'|'TG'|'TH'|'TJ'|'TK'|'TL'|'TM'|'TN'|'TO'|'TR'|'TT'|'TV'|'TW'|'TZ'|'UA'|'UG'|'US'|'UY'|'UZ'|'VA'|'VC'|'VE'|'VG'|'VI'|'VN'|'VU'|'WF'|'WS'|'YE'|'YT'|'ZA'|'ZM'|'ZW',
            'ZipCode': 'string',
            'PhoneNumber': 'string',
            'Email': 'string',
            'Fax': 'string',
            'ExtraParams': [
                {
                    'Name': 'DUNS_NUMBER'|'BRAND_NUMBER'|'BIRTH_DEPARTMENT'|'BIRTH_DATE_IN_YYYY_MM_DD'|'BIRTH_COUNTRY'|'BIRTH_CITY'|'DOCUMENT_NUMBER'|'AU_ID_NUMBER'|'AU_ID_TYPE'|'CA_LEGAL_TYPE'|'CA_BUSINESS_ENTITY_TYPE'|'ES_IDENTIFICATION'|'ES_IDENTIFICATION_TYPE'|'ES_LEGAL_FORM'|'FI_BUSINESS_NUMBER'|'FI_ID_NUMBER'|'IT_PIN'|'RU_PASSPORT_DATA'|'SE_ID_NUMBER'|'SG_ID_NUMBER'|'VAT_NUMBER',
                    'Value': 'string'
                },
            ]
        },
        PrivacyProtectAdminContact=True|False,
        PrivacyProtectRegistrantContact=True|False,
        PrivacyProtectTechContact=True|False
    )
    
    
    :type DomainName: string
    :param DomainName: [REQUIRED]
            The name of the domain that you want to transfer to Amazon Route 53.
            Constraints: The domain name can contain only the letters a through z, the numbers 0 through 9, and hyphen (-). Internationalized Domain Names are not supported.
            

    :type IdnLangCode: string
    :param IdnLangCode: Reserved for future use.

    :type DurationInYears: integer
    :param DurationInYears: [REQUIRED]
            The number of years that you want to register the domain for. Domains are registered for a minimum of one year. The maximum period depends on the top-level domain.
            Default: 1
            

    :type Nameservers: list
    :param Nameservers: Contains details for the host and glue IP addresses.
            (dict) --Nameserver includes the following elements.
            Name (string) -- [REQUIRED]The fully qualified host name of the name server.
            Constraint: Maximum 255 characters
            GlueIps (list) --Glue IP address of a name server entry. Glue IP addresses are required only when the name of the name server is a subdomain of the domain. For example, if your domain is example.com and the name server for the domain is ns.example.com, you need to specify the IP address for ns.example.com.
            Constraints: The list can contain only one IPv4 and one IPv6 address.
            (string) --
            
            

    :type AuthCode: string
    :param AuthCode: The authorization code for the domain. You get this value from the current registrar.

    :type AutoRenew: boolean
    :param AutoRenew: Indicates whether the domain will be automatically renewed (true) or not (false). Autorenewal only takes effect after the account is charged.
            Default: true
            

    :type AdminContact: dict
    :param AdminContact: [REQUIRED]
            Provides detailed contact information.
            FirstName (string) --First name of contact.
            LastName (string) --Last name of contact.
            ContactType (string) --Indicates whether the contact is a person, company, association, or public organization. If you choose an option other than PERSON , you must enter an organization name, and you can't enable privacy protection for the contact.
            OrganizationName (string) --Name of the organization for contact types other than PERSON .
            AddressLine1 (string) --First line of the contact's address.
            AddressLine2 (string) --Second line of contact's address, if any.
            City (string) --The city of the contact's address.
            State (string) --The state or province of the contact's city.
            CountryCode (string) --Code for the country of the contact's address.
            ZipCode (string) --The zip or postal code of the contact's address.
            PhoneNumber (string) --The phone number of the contact.
            Constraints: Phone number must be specified in the format '+[country dialing code].[number including any area code]'. For example, a US phone number might appear as '+1.1234567890' .
            Email (string) --Email address of the contact.
            Fax (string) --Fax number of the contact.
            Constraints: Phone number must be specified in the format '+[country dialing code].[number including any area code]'. For example, a US phone number might appear as '+1.1234567890' .
            ExtraParams (list) --A list of name-value pairs for parameters required by certain top-level domains.
            (dict) --ExtraParam includes the following elements.
            Name (string) -- [REQUIRED]Name of the additional parameter required by the top-level domain.
            Value (string) -- [REQUIRED]Values corresponding to the additional parameter names required by some top-level domains.
            
            

    :type RegistrantContact: dict
    :param RegistrantContact: [REQUIRED]
            Provides detailed contact information.
            FirstName (string) --First name of contact.
            LastName (string) --Last name of contact.
            ContactType (string) --Indicates whether the contact is a person, company, association, or public organization. If you choose an option other than PERSON , you must enter an organization name, and you can't enable privacy protection for the contact.
            OrganizationName (string) --Name of the organization for contact types other than PERSON .
            AddressLine1 (string) --First line of the contact's address.
            AddressLine2 (string) --Second line of contact's address, if any.
            City (string) --The city of the contact's address.
            State (string) --The state or province of the contact's city.
            CountryCode (string) --Code for the country of the contact's address.
            ZipCode (string) --The zip or postal code of the contact's address.
            PhoneNumber (string) --The phone number of the contact.
            Constraints: Phone number must be specified in the format '+[country dialing code].[number including any area code]'. For example, a US phone number might appear as '+1.1234567890' .
            Email (string) --Email address of the contact.
            Fax (string) --Fax number of the contact.
            Constraints: Phone number must be specified in the format '+[country dialing code].[number including any area code]'. For example, a US phone number might appear as '+1.1234567890' .
            ExtraParams (list) --A list of name-value pairs for parameters required by certain top-level domains.
            (dict) --ExtraParam includes the following elements.
            Name (string) -- [REQUIRED]Name of the additional parameter required by the top-level domain.
            Value (string) -- [REQUIRED]Values corresponding to the additional parameter names required by some top-level domains.
            
            

    :type TechContact: dict
    :param TechContact: [REQUIRED]
            Provides detailed contact information.
            FirstName (string) --First name of contact.
            LastName (string) --Last name of contact.
            ContactType (string) --Indicates whether the contact is a person, company, association, or public organization. If you choose an option other than PERSON , you must enter an organization name, and you can't enable privacy protection for the contact.
            OrganizationName (string) --Name of the organization for contact types other than PERSON .
            AddressLine1 (string) --First line of the contact's address.
            AddressLine2 (string) --Second line of contact's address, if any.
            City (string) --The city of the contact's address.
            State (string) --The state or province of the contact's city.
            CountryCode (string) --Code for the country of the contact's address.
            ZipCode (string) --The zip or postal code of the contact's address.
            PhoneNumber (string) --The phone number of the contact.
            Constraints: Phone number must be specified in the format '+[country dialing code].[number including any area code]'. For example, a US phone number might appear as '+1.1234567890' .
            Email (string) --Email address of the contact.
            Fax (string) --Fax number of the contact.
            Constraints: Phone number must be specified in the format '+[country dialing code].[number including any area code]'. For example, a US phone number might appear as '+1.1234567890' .
            ExtraParams (list) --A list of name-value pairs for parameters required by certain top-level domains.
            (dict) --ExtraParam includes the following elements.
            Name (string) -- [REQUIRED]Name of the additional parameter required by the top-level domain.
            Value (string) -- [REQUIRED]Values corresponding to the additional parameter names required by some top-level domains.
            
            

    :type PrivacyProtectAdminContact: boolean
    :param PrivacyProtectAdminContact: Whether you want to conceal contact information from WHOIS queries. If you specify true , WHOIS ('who is') queries will return contact information for our registrar partner, Gandi, instead of the contact information that you enter.
            Default: true
            

    :type PrivacyProtectRegistrantContact: boolean
    :param PrivacyProtectRegistrantContact: Whether you want to conceal contact information from WHOIS queries. If you specify true , WHOIS ('who is') queries will return contact information for our registrar partner, Gandi, instead of the contact information that you enter.
            Default: true
            

    :type PrivacyProtectTechContact: boolean
    :param PrivacyProtectTechContact: Whether you want to conceal contact information from WHOIS queries. If you specify true , WHOIS ('who is') queries will return contact information for our registrar partner, Gandi, instead of the contact information that you enter.
            Default: true
            

    :rtype: dict
    :return: {
        'OperationId': 'string'
    }
    
    
    """
    pass