def connect_to_database(host=None, port=None, connect=False, **kwargs):
    """
    Explicitly begins a database connection for the application
    (if this function is not called, a connection is created when
    it is first needed).  Takes arguments identical to
    pymongo.MongoClient.__init__
    
    @param host: the hostname to connect to
    @param port: the port to connect to
    @param connect:  if True, immediately begin connecting to MongoDB in the
        background; otherwise connect on the first operation
    """
    return CONNECTION.connect(host=host, port=port, connect=connect, **kwargs)