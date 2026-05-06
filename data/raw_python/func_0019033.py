def get_modelinsertion(cls, model, indent) -> str:
        """Return the insertion string required for the given application model.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> from hydpy import prepare_model
        >>> model = prepare_model('hland_v1')
        >>> print(XSDWriter.get_modelinsertion(model, 1))   # doctest: +ELLIPSIS
            <element name="inputs"
                     minOccurs="0">
                <complexType>
                    <sequence>
                        <element
                            name="p"
                            minOccurs="0"/>
        ...
            </element>
            <element name="fluxes"
                     minOccurs="0">
        ...
            </element>
            <element name="states"
                     minOccurs="0">
        ...
            </element>
        """
        texts = []
        for name in ('inputs', 'fluxes', 'states'):
            subsequences = getattr(model.sequences, name, None)
            if subsequences:
                texts.append(
                    cls.get_subsequencesinsertion(subsequences, indent))
        return '\n'.join(texts)