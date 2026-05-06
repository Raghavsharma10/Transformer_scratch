def marshall_lines(self, collector):
        """ Marshalls a collector and returns the storage/transfer format in
            a tuple, this tuple has reprensentation format per element.
        """

        if isinstance(collector, collectors.Counter):
            exec_method = self._format_counter
        elif isinstance(collector, collectors.Gauge):
            exec_method = self._format_gauge
        elif isinstance(collector, collectors.Summary):
            exec_method = self._format_summary
        else:
            raise TypeError("Not a valid object format")

        # create headers
        help_header = TextFormat.HELP_FMT.format(name=collector.name,
                                                 help_text=collector.help_text)

        type_header = TextFormat.TYPE_FMT.format(name=collector.name,
                                                 value_type=collector.REPR_STR)

        # Prepare start headers
        lines = [help_header, type_header]

        for i in collector.get_all():
            r = exec_method(i, collector.name, collector.const_labels)

            # Check if it returns one or multiple lines
            if not isinstance(r, str) and isinstance(r, collections.Iterable):
                lines.extend(r)
            else:
                lines.append(r)

        return lines