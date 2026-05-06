def perform_batch_reply(
            self,
            *,
            callback: Callable[..., str]=None,
            target_handles: Dict[str, str]=None,
            lookback_limit: int=20,
            per_service_lookback_limit: Dict[str, int]=None,
    ) -> IterationRecord:
        """
        Performs batch reply on target accounts.
        Looks up the recent messages of the target user,
        applies the callback,
        and replies with
        what the callback generates.

        :param callback: a callback taking a message id,
            message contents,
            and optional extra keys,
            and returning a message string.
        :param targets: a dictionary of service names to target handles
            (currently only one per service).
        :param lookback_limit: a lookback limit of how many messages to consider (optional).
        :param per_service_lookback: and a dictionary of service names to per-service
            lookback limits.
            takes preference over lookback_limit (optional).
        :returns: new record of iteration
        :raises BotSkeletonException: raises BotSkeletonException if batch reply fails or cannot be
            performed
        """
        if callback is None:
            raise BotSkeletonException("Callback must be provided.""")

        if target_handles is None:
            raise BotSkeletonException("Targets must be provided.""")

        if lookback_limit > self.lookback_limit:
            raise BotSkeletonException(
                f"Lookback_limit cannot exceed {self.lookback_limit}, " +
                f"but it was {lookback_limit}"
            )

        # use per-service lookback dict for convenience in a moment.
        # if necessary, use lookback_limit to fill it out.
        lookback_dict = per_service_lookback_limit
        if (lookback_dict is None):
            lookback_dict = {}

        record = IterationRecord(extra_keys=self.extra_keys)
        for key, output in self.outputs.items():
            if key not in lookback_dict:
                lookback_dict[key] = lookback_limit

            if target_handles.get(key, None) is None:
                self.log.info(f"No target for output {key}, skipping this output.")

            elif not output.get("active", False):
                self.log.info(f"Output {key} is inactive. Not calling batch reply.")

            elif output["active"]:
                self.log.info(f"Output {key} is active, calling batch reply on it.")
                entry: Any = output["obj"]
                output_result = entry.perform_batch_reply(callback=callback,
                                                          target_handle=target_handles[key],
                                                          lookback_limit=lookback_dict[key],
                                                          )
                record.output_records[key] = output_result

        self.history.append(record)
        self.update_history()

        return record