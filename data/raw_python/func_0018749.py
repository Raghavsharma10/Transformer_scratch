def execute(self, identity_records: 'RDD', old_state_rdd: Optional['RDD'] = None) -> 'RDD':
        """
        Executes Blurr BTS with the given records. old_state_rdd can be provided to load an older
        state from a previous run.

        :param identity_records: RDD of the form Tuple[Identity, List[TimeAndRecord]]
        :param old_state_rdd: A previous streaming BTS state RDD as Tuple[Identity, Streaming BTS
            State]
        :return: RDD[Identity, Tuple[Streaming BTS State, List of Window BTS output]]
        """
        identity_records_with_state = identity_records
        if old_state_rdd:
            identity_records_with_state = identity_records.fullOuterJoin(old_state_rdd)
        return identity_records_with_state.map(lambda x: self._execute_per_identity_records(x))