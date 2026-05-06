def agent(agent_id):
    '''
    Show the information about the given agent.
    '''
    fields = [
        ('ID', 'id'),
        ('Status', 'status'),
        ('Region', 'region'),
        ('First Contact', 'first_contact'),
        ('CPU Usage (%)', 'cpu_cur_pct'),
        ('Used Memory (MiB)', 'mem_cur_bytes'),
        ('Total slots', 'available_slots'),
        ('Occupied slots', 'occupied_slots'),
    ]
    if is_legacy_server():
        del fields[9]
        del fields[6]
    q = 'query($agent_id:String!) {' \
        '  agent(agent_id:$agent_id) { $fields }' \
        '}'
    q = q.replace('$fields', ' '.join(item[1] for item in fields))
    v = {'agent_id': agent_id}
    with Session() as session:
        try:
            resp = session.Admin.query(q, v)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        info = resp['agent']
        rows = []
        for name, key in fields:
            if key == 'mem_cur_bytes' and info[key] is not None:
                info[key] = round(info[key] / 2 ** 20, 1)
            if key in info:
                rows.append((name, info[key]))
        print(tabulate(rows, headers=('Field', 'Value')))