def handle(conn, addr, gateway, *args, **kwargs):
    """
    NOTE: use tcp instead of udp because some operations need ack
    """
    conn.sendall(b'OK pubsub 1.0\n')
    while True:
        try:
            s = conn.recv(1024).decode('utf-8').strip()
            if not s:
                conn.close()
                break
        except ConnectionResetError:
            logger.debug('Client close the connection.')
            break

        parts = s.split(' ')
        if len(parts) != 2:
            conn.send(b"Invalid command\n")
            continue
        cmd, topic = parts
        if cmd.lower() != 'sub':
            conn.send(bytes("Unknown command '{}'\n".format(cmd.lower()), 'utf-8'))
            continue
        if topic not in gateway.topics:
            conn.send(bytes("Unknown topic '{}'\n".format(topic), 'utf-8'))
            continue
        conn.sendall(bytes('ACK {} {}\n'.format(cmd, topic), 'utf-8'))
        subscriber = Subscriber(addr, conn)
        gateway.link(topic, subscriber)
        break