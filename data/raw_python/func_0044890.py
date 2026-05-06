def connect(self, server):
        "Connects to a server and return a connection id."
        if 'connections' not in session:
            session['connections'] = {}
            session.save()

        conns = session['connections']
        id = str(len(conns))
        conn = Connection(server)
        conns[id] = conn
        yield request.environ['cogen.core'].events.AddCoro(conn.pull)
        yield id