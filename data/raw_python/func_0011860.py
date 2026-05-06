def send_with_many_media(
            self,
            *args: str,
            text: str=None,
            files: List[str]=None,
            captions: List[str]=[],
    ) -> IterationRecord:
        """
        Post with several media.
        Provide filenames so outputs can handle their own uploads.

        :param args: positional arguments.
            expected:
                text to send as message in post.
                files to be uploaded.
                captions to be paired with files.
            keyword arguments preferred over positional ones.
        :param text: text to send as message in post.
        :param files: files to be uploaded in post.
        :param captions: captions to be uploaded alongside files.
        :returns: new record of iteration
        """
        if text is None:
            if len(args) < 1:
                raise TypeError(("Please provide either required positional argument "
                                 "TEXT, or keyword argument text=TEXT"))
            else:
                final_text = args[0]
        else:
            final_text = text

        if files is None:
            if len(args) < 2:
                raise TypeError(("Please provide either positional argument "
                                 "FILES, or keyword argument files=FILES"))
            else:
                final_files = list(args[1:])
        else:
            final_files = files

        # captions have never been permitted to be provided as positional args
        # (kind of backed myself into that)
        # so they just get defaulted and it's fine.

        record = IterationRecord(extra_keys=self.extra_keys)
        for key, output in self.outputs.items():
            if output["active"]:
                self.log.info(f"Output {key} is active, calling media send on it.")
                entry: Any = output["obj"]
                output_result = entry.send_with_media(text=final_text,
                                                      files=final_files,
                                                      captions=captions)
                record.output_records[key] = output_result
            else:
                self.log.info(f"Output {key} is inactive. Not sending with media.")

        self.history.append(record)
        self.update_history()

        return record