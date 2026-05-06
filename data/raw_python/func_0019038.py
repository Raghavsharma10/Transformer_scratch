def get_itemtypesinsertion(cls, itemgroup, indent) -> str:
        """Return a string defining the required types for the given
        exchange item group.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_itemtypesinsertion(
        ...     'setitems', 1))    # doctest: +ELLIPSIS
            <complexType name="arma_v1_setitemsType">
        ...
            </complexType>
        <BLANKLINE>
            <complexType name="dam_v001_setitemsType">
        ...
            <complexType name="nodes_setitemsType">
        ...
        """
        subs = []
        for modelname in cls.get_modelnames():
            subs.append(cls.get_itemtypeinsertion(itemgroup, modelname, indent))
        subs.append(cls.get_nodesitemtypeinsertion(itemgroup, indent))
        return '\n'.join(subs)