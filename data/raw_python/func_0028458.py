def status():
    '''Show the manager's current status.'''
    with Session() as session:
        resp = session.Manager.status()
        print(tabulate([('Status', 'Active Sessions'),
                        (resp['status'], resp['active_sessions'])],
                       headers='firstrow'))