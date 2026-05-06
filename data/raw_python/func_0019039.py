def get_itemtypeinsertion(cls, itemgroup, modelname, indent) -> str:
        """Return a string defining the required types for the given
        combination of an exchange item group and an application model.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_itemtypeinsertion(
        ...     'setitems', 'hland_v1', 1))    # doctest: +ELLIPSIS
            <complexType name="hland_v1_setitemsType">
                <sequence>
                    <element ref="hpcb:selections"
                             minOccurs="0"/>
                    <element ref="hpcb:devices"
                             minOccurs="0"/>
                    <element name="control"
                             minOccurs="0"
                             maxOccurs="unbounded">
        ...
                </sequence>
            </complexType>
        <BLANKLINE>
        """
        blanks = ' ' * (indent * 4)
        type_ = cls._get_itemstype(modelname, itemgroup)
        subs = [
            f'{blanks}<complexType name="{type_}">',
            f'{blanks}    <sequence>',
            f'{blanks}        <element ref="hpcb:selections"',
            f'{blanks}                 minOccurs="0"/>',
            f'{blanks}        <element ref="hpcb:devices"',
            f'{blanks}                 minOccurs="0"/>',
            cls.get_subgroupsiteminsertion(itemgroup, modelname, indent+2),
            f'{blanks}    </sequence>',
            f'{blanks}</complexType>',
            f'']
        return '\n'.join(subs)