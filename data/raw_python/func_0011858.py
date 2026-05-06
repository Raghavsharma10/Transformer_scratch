def send(
            self,
            *args: str,
            text: str=None,
    ) -> IterationRecord:
        """
        Post text-only to all outputs.

        :param args: positional arguments.
            expected: text to send as message in post.
            keyword text argument is preferred over this.
        :param text: text to send as message in post.
        :returns: new record of iteration
        """
        if text is not None:
            final_text = text
        else:
            if len(args) == 0:
                raise BotSkeletonException(("Please provide text either as a positional arg or "
                                            "as a keyword arg (text=TEXT)"))
            else:
                final_text = args[0]

        # TODO there could be some annotation stuff here.
        record = IterationRecord(extra_keys=self.extra_keys)
        for key, output in self.outputs.items():
            if output["active"]:
                self.log.info(f"Output {key} is active, calling send on it.")
                entry: Any = output["obj"]
                output_result = entry.send(text=final_text)
                record.output_records[key] = output_result

            else:
                self.log.info(f"Output {key} is inactive. Not sending.")

        self.history.append(record)
        self.update_history()

        return record