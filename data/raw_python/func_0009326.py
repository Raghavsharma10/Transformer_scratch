def compare_wfns(cls, source, target):
        """
        Compares two WFNs and returns a generator of pairwise attribute-value
        comparison results. It provides full access to the individual
        comparison results to enable use-case specific implementations
        of novel name-comparison algorithms.

        Compare each attribute of the Source WFN to the Target WFN:

        :param CPE2_3_WFN source: first WFN CPE Name
        :param CPE2_3_WFN target: seconds WFN CPE Name
        :returns: generator of pairwise attribute comparison results
        :rtype: generator
        """

        # Compare results using the get() function in WFN
        for att in CPEComponent.CPE_COMP_KEYS_EXTENDED:
            value_src = source.get_attribute_values(att)[0]
            if value_src.find('"') > -1:
                # Not a logical value: del double quotes
                value_src = value_src[1:-1]

            value_tar = target.get_attribute_values(att)[0]
            if value_tar.find('"') > -1:
                # Not a logical value: del double quotes
                value_tar = value_tar[1:-1]

            yield (att, CPESet2_3._compare(value_src, value_tar))