def get_nodesitemtypeinsertion(cls, itemgroup, indent) -> str:
        """Return a string defining the required types for the given
        combination of an exchange item group and |Node| objects.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_nodesitemtypeinsertion(
        ...     'setitems', 1))    # doctest: +ELLIPSIS
            <complexType name="nodes_setitemsType">
                <sequence>
                    <element ref="hpcb:selections"
                             minOccurs="0"/>
                    <element ref="hpcb:devices"
                             minOccurs="0"/>
                    <element name="sim"
                             type="hpcb:setitemType"
                             minOccurs="0"
                             maxOccurs="unbounded"/>
                    <element name="obs"
                             type="hpcb:setitemType"
                             minOccurs="0"
                             maxOccurs="unbounded"/>
                    <element name="sim.series"
                             type="hpcb:setitemType"
                             minOccurs="0"
                             maxOccurs="unbounded"/>
                    <element name="obs.series"
                             type="hpcb:setitemType"
                             minOccurs="0"
                             maxOccurs="unbounded"/>
                </sequence>
            </complexType>
        <BLANKLINE>
        """
        blanks = ' ' * (indent * 4)
        subs = [
            f'{blanks}<complexType name="nodes_{itemgroup}Type">',
            f'{blanks}    <sequence>',
            f'{blanks}        <element ref="hpcb:selections"',
            f'{blanks}                 minOccurs="0"/>',
            f'{blanks}        <element ref="hpcb:devices"',
            f'{blanks}                 minOccurs="0"/>']
        type_ = 'getitemType' if itemgroup == 'getitems' else 'setitemType'
        for name in ('sim', 'obs', 'sim.series', 'obs.series'):
            subs.extend([
                f'{blanks}        <element name="{name}"',
                f'{blanks}                 type="hpcb:{type_}"',
                f'{blanks}                 minOccurs="0"',
                f'{blanks}                 maxOccurs="unbounded"/>'])
        subs.extend([
            f'{blanks}    </sequence>',
            f'{blanks}</complexType>',
            f''])
        return '\n'.join(subs)