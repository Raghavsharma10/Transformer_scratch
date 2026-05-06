def _compare(cls, source, target):
        """
        Compares two values associated with a attribute of two WFNs,
        which may be logical values (ANY or NA) or string values.

        :param string source: First attribute value
        :param string target: Second attribute value
        :returns: The attribute comparison relation.
        :rtype: int

        This function is a support function for compare_WFNs.
        """

        if (CPESet2_3._is_string(source)):
            source = source.lower()
        if (CPESet2_3._is_string(target)):
            target = target.lower()

        # In this specification, unquoted wildcard characters in the target
        # yield an undefined result

        if (CPESet2_3._is_string(target) and
           CPESet2_3._contains_wildcards(target)):
            return CPESet2_3.LOGICAL_VALUE_UNDEFINED

        # If source and target attribute values are equal,
        # then the result is EQUAL
        if (source == target):
            return CPESet2_3.LOGICAL_VALUE_EQUAL

        # If source attribute value is ANY, then the result is SUPERSET
        if (source == CPEComponent2_3_WFN.VALUE_ANY):
            return CPESet2_3.LOGICAL_VALUE_SUPERSET

        # If target attribute value is ANY, then the result is SUBSET
        if (target == CPEComponent2_3_WFN.VALUE_ANY):
            return CPESet2_3.LOGICAL_VALUE_SUBSET

        # If either source or target attribute value is NA
        # then the result is DISJOINT
        isSourceNA = source == CPEComponent2_3_WFN.VALUE_NA
        isTargetNA = target == CPEComponent2_3_WFN.VALUE_NA

        if (isSourceNA or isTargetNA):
            return CPESet2_3.LOGICAL_VALUE_DISJOINT

        # If we get to this point, we are comparing two strings
        return CPESet2_3._compare_strings(source, target)