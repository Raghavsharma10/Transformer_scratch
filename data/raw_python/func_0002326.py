def get_output(self, include_exceptions=False):
        """
        Return the output in the correct ordering.
        :rtype: list[Tuple[contentitem, O]]
        """
        # Order all rendered items in the correct sequence.
        # Don't assume the derived tables are in perfect shape, hence the dict + KeyError handling.
        # The derived tables could be truncated/reset or store_output() could be omitted.
        ordered_output = []
        for item_id in self.output_ordering:
            contentitem = self.item_source[item_id]
            try:
                output = self.item_output[item_id]
            except KeyError:
                # The item was not rendered!
                if not include_exceptions:
                    continue

                output = self.MISSING
            else:
                # Filter exceptions out.
                if not include_exceptions:
                    if isinstance(output, Exception) or output is self.SKIPPED:
                        continue

            ordered_output.append((contentitem, output))

        return ordered_output