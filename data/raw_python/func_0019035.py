def get_exchangeinsertion(cls):
        """Return the complete string related to the definition of exchange
        items to be inserted into the string of the template file.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_exchangeinsertion())    # doctest: +ELLIPSIS
            <complexType name="arma_v1_mathitemType">
        ...
            <element name="setitems">
        ...
            <complexType name="arma_v1_setitemsType">
        ...
            <element name="additems">
        ...
            <element name="getitems">
        ...
        """
        indent = 1
        subs = [cls.get_mathitemsinsertion(indent)]
        for groupname in ('setitems', 'additems', 'getitems'):
            subs.append(cls.get_itemsinsertion(groupname, indent))
            subs.append(cls.get_itemtypesinsertion(groupname, indent))
        return '\n'.join(subs)