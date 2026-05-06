def flush_read_tuple(self):
        """Flush the internal buffer of reads.
		"""
        if not self.is_empty():
            suffix_comment_buffer = []
            if self._info_simulator is not None:
                suffix_comment_buffer.append(self._info_simulator)
            if self._info_reads_in_tuple:
                # todo: orientation (FF, FR, etc.)
                # orientation="".join([])
                suffix_comment_buffer.append("reads-in-tuple:{}".format(len(self.seqs_bases)))
            if len(suffix_comment_buffer) != 0:
                suffix_comment = "[{}]".format(",".join(suffix_comment_buffer))
            else:
                suffix_comment = ""

            rnf_name = self._rnf_profile.get_rnf_name(
                rnftools.rnfformat.ReadTuple(
                    segments=self.segments,
                    read_tuple_id=self.current_read_tuple_id,
                    suffix=suffix_comment,
                )
            )
            fq_reads = [
                os.linesep.join(
                    [
                        "@{rnf_name}{read_suffix}".format(
                            rnf_name=rnf_name,
                            read_suffix="/{}".format(str(i + 1)) if len(self.seqs_bases) > 1 else "",
                        ),
                        self.seqs_bases[i],
                        "+",
                        self.seqs_qualities[i],
                    ]
                ) for i in range(len(self.seqs_bases))
            ]
            self._fq_file.write(os.linesep.join(fq_reads))
            self._fq_file.write(os.linesep)
            self.empty()