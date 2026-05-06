def _pfp__process_metadata(self):
        """Process the metadata once the entire struct has been
        declared.
        """
        if self._pfp__metadata_processor is None:
            return

        metadata_info = self._pfp__metadata_processor()
        if isinstance(metadata_info, list):
            for metadata in metadata_info:
                if metadata["type"] == "watch":
                    self._pfp__set_watch(
                        metadata["watch_fields"],
                        metadata["update_func"],
                        *metadata["func_call_info"]
                    )

                elif metadata["type"] == "packed":
                    del metadata["type"]
                    self._pfp__set_packer(**metadata)
                    if self._pfp__can_unpack():
                        self._pfp__unpack_data(self.raw_data)