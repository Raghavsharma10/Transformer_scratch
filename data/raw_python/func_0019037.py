def get_itemsinsertion(cls, itemgroup, indent) -> str:
        """Return a string defining the XML element for the given
        exchange item group.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_itemsinsertion(
        ...     'setitems', 1))    # doctest: +ELLIPSIS
            <element name="setitems">
                <complexType>
                    <sequence>
                        <element ref="hpcb:selections"
                                 minOccurs="0"/>
                        <element ref="hpcb:devices"
                                 minOccurs="0"/>
        ...
                        <element name="hland_v1"
                                 type="hpcb:hland_v1_setitemsType"
                                 minOccurs="0"
                                 maxOccurs="unbounded"/>
        ...
                        <element name="nodes"
                                 type="hpcb:nodes_setitemsType"
                                 minOccurs="0"
                                 maxOccurs="unbounded"/>
                    </sequence>
                    <attribute name="info" type="string"/>
                </complexType>
            </element>
        <BLANKLINE>
        """
        blanks = ' ' * (indent*4)
        subs = []
        subs.extend([
            f'{blanks}<element name="{itemgroup}">',
            f'{blanks}    <complexType>',
            f'{blanks}        <sequence>',
            f'{blanks}            <element ref="hpcb:selections"',
            f'{blanks}                     minOccurs="0"/>',
            f'{blanks}            <element ref="hpcb:devices"',
            f'{blanks}                     minOccurs="0"/>'])
        for modelname in cls.get_modelnames():
            type_ = cls._get_itemstype(modelname, itemgroup)
            subs.append(f'{blanks}            <element name="{modelname}"')
            subs.append(f'{blanks}                     type="hpcb:{type_}"')
            subs.append(f'{blanks}                     minOccurs="0"')
            subs.append(f'{blanks}                     maxOccurs="unbounded"/>')
        if itemgroup in ('setitems', 'getitems'):
            type_ = f'nodes_{itemgroup}Type'
            subs.append(f'{blanks}            <element name="nodes"')
            subs.append(f'{blanks}                     type="hpcb:{type_}"')
            subs.append(f'{blanks}                     minOccurs="0"')
            subs.append(f'{blanks}                     maxOccurs="unbounded"/>')
        subs.extend([
                f'{blanks}        </sequence>',
                f'{blanks}        <attribute name="info" type="string"/>',
                f'{blanks}    </complexType>',
                f'{blanks}</element>',
                f''])
        return '\n'.join(subs)