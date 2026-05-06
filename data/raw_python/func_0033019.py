def results(self, trial_ids):
        """
        Accepts a sequence of trial ids and returns a pandas dataframe
        with the schema

        trial_id, iteration?, *metric_schema_union

        where iteration is an optional column that specifies the iteration
        when a user logged a metric, if the user supplied one. The iteration
        column is added if any metric was logged with an iteration.
        Then, every metric name that was ever logged is a column in the
        metric_schema_union.
        """
        metadata_folder = os.path.join(self.log_dir, constants.METADATA_FOLDER)
        dfs = []
        # TODO: various file-creation corner cases like the result file not
        # always existing if stuff is not logged and etc should be ironed out
        # (would probably be easier if we had a centralized Sync class which
        # relied on some formal remote store semantics).
        for trial_id in trial_ids:
            # TODO constants should just contain the recipes for filename
            # construction instead of this multi-file implicit constraint
            result_file = os.path.join(
                metadata_folder, trial_id + "_" + constants.RESULT_SUFFIX)
            assert os.path.isfile(result_file), result_file
            dfs.append(pd.read_json(result_file, typ='frame', lines=True))
        df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
        return df