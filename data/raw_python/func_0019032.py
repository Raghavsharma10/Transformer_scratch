def get_insertion(cls) -> str:
        """Return the complete string to be inserted into the string of the
        template file.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_insertion())    # doctest: +ELLIPSIS
             <element name="arma_v1"
                      substitutionGroup="hpcb:sequenceGroup"
                      type="hpcb:arma_v1Type"/>
        <BLANKLINE>
             <complexType name="arma_v1Type">
                 <complexContent>
                     <extension base="hpcb:sequenceGroupType">
                         <sequence>
                            <element name="fluxes"
                                     minOccurs="0">
                                <complexType>
                                    <sequence>
                                        <element
                                            name="qin"
                                            minOccurs="0"/>
        ...
                                </complexType>
                            </element>
                         </sequence>
                     </extension>
                 </complexContent>
             </complexType>
        <BLANKLINE>
        """
        indent = 1
        blanks = ' ' * (indent+4)
        subs = []
        for name in cls.get_modelnames():
            subs.extend([
                f'{blanks}<element name="{name}"',
                f'{blanks}         substitutionGroup="hpcb:sequenceGroup"',
                f'{blanks}         type="hpcb:{name}Type"/>',
                f'',
                f'{blanks}<complexType name="{name}Type">',
                f'{blanks}    <complexContent>',
                f'{blanks}        <extension base="hpcb:sequenceGroupType">',
                f'{blanks}            <sequence>'])
            model = importtools.prepare_model(name)
            subs.append(cls.get_modelinsertion(model, indent + 4))
            subs.extend([
                f'{blanks}            </sequence>',
                f'{blanks}        </extension>',
                f'{blanks}    </complexContent>',
                f'{blanks}</complexType>',
                f''
            ])
        return '\n'.join(subs)