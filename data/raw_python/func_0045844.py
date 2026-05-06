def initialize(self, runtime):
        """Initialize this listener. Finds most recent timestamp"""
        if self.is_alive():
            raise IllegalState('notification thread is already initialized')
        if not JSON_CLIENT.is_json_client_set() and runtime is not None:
            JSON_CLIENT.set_json_client(runtime)
        try:
            cursor = JSON_CLIENT.json_client['local']['oplog.rs'].find().sort('ts', DESCENDING).limit(-1)
        except TypeError:
            # filesystem, so .json_client is a bool and not iterable
            pass
        else:
            try:
                self.last_timestamp = cursor.next()['ts']
            except StopIteration:
                self.last_timestamp = Timestamp(0, 0)