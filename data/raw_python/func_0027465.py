def create_user_dci(access_token):
    """Create the a dci user.
    username=dci, password=dci, email=dci@distributed-ci.io"""
    user_data = {'username': 'dci',
                 'email': 'dci@distributed-ci.io',
                 'enabled': True,
                 'emailVerified': True,
                 'credentials': [{'type': 'password',
                                  'value': 'dci'}]}
    r = requests.post('http://keycloak:8080/auth/admin/realms/dci-test/users',
                      data=json.dumps(user_data),
                      headers=get_auth_headers(access_token))
    if r.status_code in (201, 409):
        print('Keycloak user dci created successfully.')
    else:
        raise Exception('Error while creating user dci:\nstatus code %s\n'
                        'error: %s' % (r.status_code, r.content))