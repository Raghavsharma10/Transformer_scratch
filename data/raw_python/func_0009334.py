def language_match(self, cpeset, cpel_dom=None):
        """
        Accepts a set of known CPE Names and an expression in the CPE language,
        and delivers the answer True if the expression matches with the set.
        Otherwise, it returns False.

        :param CPELanguage self: An expression in the CPE Applicability
            Language, represented as the XML infoset for the platform element.
        :param CPESet cpeset: CPE set object to match with self expression.
        :param string cpel_dom: An expression in the CPE Applicability
            Language, represented as DOM tree.
        :returns: True if self expression can be satisfied by language matching
            against cpeset, False otherwise.
        :rtype: boolean
        """

        # Root element tag
        TAG_ROOT = '#document'
        # A container for child platform definitions
        TAG_PLATSPEC = 'cpe:platform-specification'

        # Information about a platform definition
        TAG_PLATFORM = 'cpe:platform'
        TAG_LOGITEST = 'cpe:logical-test'
        TAG_CPE = 'cpe:fact-ref'

        # Tag attributes
        ATT_NAME = 'name'
        ATT_OP = 'operator'
        ATT_NEGATE = 'negate'

        # Attribute values
        ATT_OP_AND = 'AND'
        ATT_OP_OR = 'OR'
        ATT_NEGATE_TRUE = 'TRUE'

        if cpel_dom is None:
            cpel_dom = self.document

        # Identify the root element
        if cpel_dom.nodeName == TAG_ROOT or cpel_dom.nodeName == TAG_PLATSPEC:
            for node in cpel_dom.childNodes:
                if node.nodeName == TAG_PLATSPEC:
                    return self.language_match(cpeset, node)
                if node.nodeName == TAG_PLATFORM:
                    return self.language_match(cpeset, node)

        # Identify a platform element
        elif cpel_dom.nodeName == TAG_PLATFORM:
            for node in cpel_dom.childNodes:
                if node.nodeName == TAG_LOGITEST:
                    return self.language_match(cpeset, node)

        # Identify a CPE element
        elif cpel_dom.nodeName == TAG_CPE:
            cpename = cpel_dom.getAttribute(ATT_NAME)
            c = CPE2_2(cpename)

            # Try to match a CPE name with CPE set
            return cpeset.name_match(c)

        # Identify a logical operator element
        elif cpel_dom.nodeName == TAG_LOGITEST:
            count = 0
            len = 0
            answer = False

            for node in cpel_dom.childNodes:
                if node.nodeName.find("#") == 0:
                    continue
                len = len + 1
                if self.language_match(cpeset, node):
                    count = count + 1

            operator = cpel_dom.getAttribute(ATT_OP).upper()

            if operator == ATT_OP_AND:
                if count == len:
                    answer = True
            elif operator == ATT_OP_OR:
                if count > 0:
                    answer = True

            operator_not = cpel_dom.getAttribute(ATT_NEGATE)
            if operator_not:
                if operator_not.upper() == ATT_NEGATE_TRUE:
                    answer = not answer

            return answer
        else:
            return False