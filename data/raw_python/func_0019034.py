def get_subsequencesinsertion(cls, subsequences, indent) -> str:
        """Return the insertion string required for the given group of
        sequences.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> from hydpy import prepare_model
        >>> model = prepare_model('hland_v1')
        >>> print(XSDWriter.get_subsequencesinsertion(
        ...     model.sequences.fluxes, 1))    # doctest: +ELLIPSIS
            <element name="fluxes"
                     minOccurs="0">
                <complexType>
                    <sequence>
                        <element
                            name="tmean"
                            minOccurs="0"/>
                        <element
                            name="tc"
                            minOccurs="0"/>
        ...
                        <element
                            name="qt"
                            minOccurs="0"/>
                    </sequence>
                </complexType>
            </element>

        """
        blanks = ' ' * (indent*4)
        lines = [f'{blanks}<element name="{subsequences.name}"',
                 f'{blanks}         minOccurs="0">',
                 f'{blanks}    <complexType>',
                 f'{blanks}        <sequence>']
        for sequence in subsequences:
            lines.append(cls.get_sequenceinsertion(sequence, indent + 3))
        lines.extend([f'{blanks}        </sequence>',
                      f'{blanks}    </complexType>',
                      f'{blanks}</element>'])
        return '\n'.join(lines)