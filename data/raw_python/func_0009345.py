def _unbind(cls, boundname):
        """
        Unbinds a bound form to a WFN.

        :param string boundname: CPE name
        :returns: WFN object associated with boundname.
        :rtype: CPE2_3_WFN
        """

        try:
            fs = CPE2_3_FS(boundname)
        except:
            # CPE name is not formatted string
            try:
                uri = CPE2_3_URI(boundname)
            except:
                # CPE name is not URI but WFN
                return CPE2_3_WFN(boundname)
            else:
                return CPE2_3_WFN(uri.as_wfn())
        else:
            return CPE2_3_WFN(fs.as_wfn())