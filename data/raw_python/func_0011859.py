def send_with_one_media(
            self,
            *args: str,
            text: str=None,
            file: str=None,
            caption: str=None,
    ) -> IterationRecord:
        """
        Post with one media item to all outputs.
        Provide filename so outputs can handle their own uploads.

        :param args: positional arguments.
            expected:
                text to send as message in post.
                file to be uploaded.
                caption to be paired with file.
            keyword arguments preferred over positional ones.
        :param text: text to send as message in post.
        :param file: file to be uploaded in post.
        :param caption: caption to be uploaded alongside file.
        :returns: new record of iteration
        """
        final_text = text
        if final_text is None:
            if len(args) < 1:
                raise TypeError(("Please provide either positional argument "
                                 "TEXT, or keyword argument text=TEXT"))
            else:
                final_text = args[0]

        final_file = file
        if final_file is None:
            if len(args) < 2:
                raise TypeError(("Please provide either positional argument "
                                            "FILE, or keyword argument file=FILE"))
            else:
                final_file = args[1]

        # this arg is ACTUALLY optional,
        # so the pattern is changed.
        final_caption = caption
        if final_caption is None:
            if len(args) >= 3:
                final_caption = args[2]

        # TODO more error checking like this.
        if final_caption is None or final_caption == "":
            captions:List[str] = []
        else:
            captions = [final_caption]

        record = IterationRecord(extra_keys=self.extra_keys)
        for key, output in self.outputs.items():
            if output["active"]:
                self.log.info(f"Output {key} is active, calling media send on it.")
                entry: Any = output["obj"]
                output_result = entry.send_with_media(text=final_text,
                                                      files=[final_file],
                                                      captions=captions)
                record.output_records[key] = output_result
            else:
                self.log.info(f"Output {key} is inactive. Not sending with media.")

        self.history.append(record)
        self.update_history()

        return record