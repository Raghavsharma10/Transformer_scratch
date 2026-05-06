def get_subgroupsiteminsertion(cls, itemgroup, modelname, indent) -> str:
        """Return a string defining the required types for the given
        combination of an exchange item group and an application model.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_subgroupsiteminsertion(
        ...     'setitems', 'hland_v1', 1))    # doctest: +ELLIPSIS
            <element name="control"
                     minOccurs="0"
                     maxOccurs="unbounded">
        ...
            </element>
            <element name="inputs"
        ...
            <element name="fluxes"
        ...
            <element name="states"
        ...
            <element name="logs"
        ...
        """
        subs = []
        model = importtools.prepare_model(modelname)
        for subvars in cls._get_subvars(model):
            subs.append(cls.get_subgroupiteminsertion(
                    itemgroup, model, subvars, indent))
        return '\n'.join(subs)