def _compare_strings(cls, source, target):
        """
        Compares a source string to a target string,
        and addresses the condition in which the source string
        includes unquoted special characters.

        It performs a simple regular expression match,
        with the assumption that (as required) unquoted special characters
        appear only at the beginning and/or the end of the source string.

        It also properly differentiates between unquoted and quoted
        special characters.

        :param string source: First string value
        :param string target: Second string value
        :returns: The comparison relation among input strings.
        :rtype: int
        """

        start = 0
        end = len(source)
        begins = 0
        ends = 0

        # Reading of initial wildcard in source
        if source.startswith(CPEComponent2_3_WFN.WILDCARD_MULTI):
            # Source starts with "*"
            start = 1
            begins = -1
        else:
            while ((start < len(source)) and
                   source.startswith(CPEComponent2_3_WFN.WILDCARD_ONE,
                                     start, start)):
                # Source starts with one or more "?"
                start += 1
                begins += 1

        # Reading of final wildcard in source
        if (source.endswith(CPEComponent2_3_WFN.WILDCARD_MULTI) and
           CPESet2_3._is_even_wildcards(source, end - 1)):

            # Source ends in "*"
            end -= 1
            ends = -1
        else:
            while ((end > 0) and
                   source.endswith(CPEComponent2_3_WFN.WILDCARD_ONE, end - 1, end) and
                   CPESet2_3._is_even_wildcards(source, end - 1)):

                # Source ends in "?"
                end -= 1
                ends += 1

        source = source[start: end]
        index = -1
        leftover = len(target)

        while (leftover > 0):
            index = target.find(source, index + 1)
            if (index == -1):
                break
            escapes = target.count("\\", 0, index)
            if ((index > 0) and (begins != -1) and
               (begins < (index - escapes))):

                break

            escapes = target.count("\\", index + 1, len(target))
            leftover = len(target) - index - escapes - len(source)
            if ((leftover > 0) and ((ends != -1) and (leftover > ends))):
                continue

            return CPESet2_3.LOGICAL_VALUE_SUPERSET

        return CPESet2_3.LOGICAL_VALUE_DISJOINT