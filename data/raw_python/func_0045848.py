def run(self):
        """main control loop for thread"""
        while True:
            try:
                cursor = JSON_CLIENT.json_client['local']['oplog.rs'].find(
                    {'ts': {'$gt': self.last_timestamp}})
            except TypeError:
                # filesystem, so .json_client is a bool and not iterable
                pass
            else:
                # http://stackoverflow.com/questions/30401063/pymongo-tailing-oplog
                cursor.add_option(2)  # tailable
                cursor.add_option(8)  # oplog_replay
                cursor.add_option(32)  # await data
                self._retry()
                for doc in cursor:
                    self.last_timestamp = doc['ts']
                    if doc['ns'] in self.receivers:
                        self._run_namespace(doc)
            time.sleep(1)