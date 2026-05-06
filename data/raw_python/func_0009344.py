def _check_fact_ref_eval(cls, cpel_dom):
        """
        Returns the result (True, False, Error) of performing the specified
        check, unless the check isnt supported, in which case it returns
        False. Error is a catch-all for all results other than True and
        False.

        :param string cpel_dom: XML infoset for the check_fact_ref element.
        :returns: result of performing the specified check
        :rtype: boolean or error
        """

        CHECK_SYSTEM = "check-system"
        CHECK_LOCATION = "check-location"
        CHECK_ID = "check-id"

        checksystemID = cpel_dom.getAttribute(CHECK_SYSTEM)
        if (checksystemID == "http://oval.mitre.org/XMLSchema/ovaldefinitions-5"):
            # Perform an OVAL check.
            # First attribute is the URI of an OVAL definitions file.
            # Second attribute is an OVAL definition ID.
            return CPELanguage2_3._ovalcheck(cpel_dom.getAttribute(CHECK_LOCATION),
                                             cpel_dom.getAttribute(CHECK_ID))

        if (checksystemID == "http://scap.nist.gov/schema/ocil/2"):
            # Perform an OCIL check.
            # First attribute is the URI of an OCIL questionnaire file.
            # Second attribute is OCIL questionnaire ID.
            return CPELanguage2_3._ocilcheck(cpel_dom.getAttribute(CHECK_LOCATION),
                                             cpel_dom.getAttribute(CHECK_ID))

        # Can add additional check systems here, with each returning a
        # True, False, or Error value
        return False