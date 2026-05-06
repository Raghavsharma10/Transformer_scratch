def check_status(self, ignore=(), status=None):
        """
        Checks status of each collection and shard to make sure that:
          a) Cluster state is active
          b) Number of docs matches across replicas for a given shard.
        Returns a dict of results for custom alerting.
        """
        self.SHARD_CHECKS = [
            {'check_msg': 'Bad Core Count Check', 'f': self._check_shard_count},
            {'check_msg': 'Bad Shard Cluster Status', 'f': self._check_shard_status}
        ]
        if status is None:
            status = self.clusterstatus()
        out = {}
        for collection in status:
            out[collection] = {}
            out[collection]['coll_status'] = True  # Means it's fine
            out[collection]['coll_messages'] = []
            for shard in status[collection]:
                self.logger.debug("Checking {}/{}".format(collection, shard))
                s_dict = status[collection][shard]
                for check in self.SHARD_CHECKS:
                    if check['check_msg'] in ignore:
                        continue
                    res = check['f'](s_dict)
                    if not res:
                        out[collection]['coll_status'] = False
                        if check['check_msg'] not in out[collection]['coll_messages']:
                            out[collection]['coll_messages'].append(check['check_msg'])
                        self.logger.debug(s_dict)
        return out