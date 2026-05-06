def load_germanet(host = None, port = None, database_name = 'germanet'):
    '''
    Loads a GermaNet instance connected to the given MongoDB instance.

    Arguments:
    - `host`: the hostname of the MongoDB instance
    - `port`: the port number of the MongoDB instance
    - `database_name`: the name of the GermaNet database on the
      MongoDB instance
    '''
    client      = MongoClient(host, port)
    germanet_db = client[database_name]
    return GermaNet(germanet_db)