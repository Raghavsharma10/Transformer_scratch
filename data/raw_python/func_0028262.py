def session(sess_id_or_alias):
    '''
    Show detailed information for a running compute session.

    SESSID: Session id or its alias.
    '''
    fields = [
        ('Session ID', 'sess_id'),
        ('Role', 'role'),
        ('Image', 'image'),
        ('Tag', 'tag'),
        ('Created At', 'created_at'),
        ('Terminated At', 'terminated_at'),
        ('Agent', 'agent'),
        ('Status', 'status'),
        ('Status Info', 'status_info'),
        ('Occupied Resources', 'occupied_slots'),
        ('CPU Used (ms)', 'cpu_used'),
        ('Used Memory (MiB)', 'mem_cur_bytes'),
        ('Max Used Memory (MiB)', 'mem_max_bytes'),
        ('Number of Queries', 'num_queries'),
        ('Network RX Bytes', 'net_rx_bytes'),
        ('Network TX Bytes', 'net_tx_bytes'),
        ('IO Read Bytes', 'io_read_bytes'),
        ('IO Write Bytes', 'io_write_bytes'),
        ('IO Max Scratch Size', 'io_max_scratch_size'),
        ('IO Current Scratch Size', 'io_cur_scratch_size'),
        ('CPU Using (%)', 'cpu_using'),
    ]
    if is_legacy_server():
        del fields[3]
    q = 'query($sess_id:String) {' \
        '  compute_session(sess_id:$sess_id) { $fields }' \
        '}'
    q = q.replace('$fields', ' '.join(item[1] for item in fields))
    v = {'sess_id': sess_id_or_alias}
    with Session() as session:
        try:
            resp = session.Admin.query(q, v)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if resp['compute_session']['sess_id'] is None:
            print('There is no such running compute session.')
            return
        print('Session detail:\n---------------')
        for i, value in enumerate(resp['compute_session'].values()):
            if fields[i][1] in ['mem_cur_bytes', 'mem_max_bytes']:
                value = round(value / 2 ** 20, 1)
            print(fields[i][0] + ': ' + str(value))