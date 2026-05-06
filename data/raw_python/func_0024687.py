def _process_generic_param(pval, def_unit, equivalencies=[]):
        """Process generic model parameter."""
        if isinstance(pval, u.Quantity):
            outval = pval.to(def_unit, equivalencies).value
        else:  # Assume already in desired unit
            outval = pval
        return outval