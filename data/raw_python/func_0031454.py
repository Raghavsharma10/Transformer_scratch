def sitesMatching(self, targets, matchCase, any_):
        """
        Find sites (i.e., sequence indices) that match a given set of target
        sequence bases.

        @param targets: A C{set} of sequence bases to look for.
        @param matchCase: If C{True}, case will be considered in matching.
        @param any_: If C{True}, return sites that match in any read. Else
            return sites that match in all reads.
        @return: A C{set} of 0-based sites that indicate where the target
            bases occur in our reads. An index will be in this set if any of
            our reads has any of the target bases in that location.
        """
        # If case is unimportant, we convert everything (target bases and
        # sequences, as we read them) to lower case.
        if not matchCase:
            targets = set(map(str.lower, targets))

        result = set() if any_ else None
        for read in self:
            sequence = read.sequence if matchCase else read.sequence.lower()
            matches = set(index for (index, base) in enumerate(sequence)
                          if base in targets)
            if any_:
                result |= matches
            else:
                if result is None:
                    result = matches
                else:
                    result &= matches
                # We can exit early if we run out of possible sites.
                if not result:
                    break

        # Make sure we don't return None.
        return result or set()