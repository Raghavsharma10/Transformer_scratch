def create_client(access_token):
    """Create the dci client in the master realm."""
    url = 'http://keycloak:8080/auth/admin/realms/dci-test/clients'
    r = requests.post(url,
                      data=json.dumps(client_data),
                      headers=get_auth_headers(access_token))
    if r.status_code in (201, 409):
        print('Keycloak client dci created successfully.')
    else:
        raise Exception(
            'Error while creating Keycloak client dci:\nstatus code %s\n'
            'error: %s' % (r.status_code, r.content)
        )