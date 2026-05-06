def load_history(self) -> List["IterationRecord"]:
        """
        Load messaging history from disk to self.

        :returns: List of iteration records comprising history.
        """
        if path.isfile(self.history_filename):
            with open(self.history_filename, "r") as f:
                try:
                    dicts = json.load(f)

                except json.decoder.JSONDecodeError as e:
                    self.log.error(f"Got error \n{e}\n decoding JSON history, overwriting it.\n"
                                   f"Former history available in {self.history_filename}.bak")
                    copyfile(self.history_filename, f"{self.history_filename}.bak")
                    return []

                history: List[IterationRecord] = []
                for hdict_pre in dicts:

                    if "_type" in hdict_pre and hdict_pre["_type"] == IterationRecord.__name__:
                        # repair any corrupted entries
                        hdict = _repair(hdict_pre)
                        record = IterationRecord.from_dict(hdict)
                        history.append(record)

                    # Be sure to handle legacy tweetrecord-only histories.
                    # Assume anything without our new _type (which should have been there from the
                    # start, whoops) is a legacy history.
                    else:
                        item = IterationRecord()

                        # Lift extra keys up to upper record (if they exist).
                        extra_keys = hdict_pre.pop("extra_keys", {})
                        item.extra_keys = extra_keys

                        hdict_obj = TweetRecord.from_dict(hdict_pre)

                        # Lift timestamp up to upper record.
                        item.timestamp = hdict_obj.timestamp

                        item.output_records["birdsite"] = hdict_obj

                        history.append(item)

                self.log.debug(f"Loaded history:\n {history}")

                return history

        else:
            return []