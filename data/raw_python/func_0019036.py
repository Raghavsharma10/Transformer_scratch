def get_mathitemsinsertion(cls, indent) -> str:
        """Return a string defining a model specific XML type extending
        `ItemType`.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_mathitemsinsertion(1))    # doctest: +ELLIPSIS
            <complexType name="arma_v1_mathitemType">
                <complexContent>
                    <extension base="hpcb:setitemType">
                        <choice>
                            <element name="control.responses"/>
        ...
                            <element name="logs.logout"/>
                        </choice>
                    </extension>
                </complexContent>
            </complexType>
        <BLANKLINE>
            <complexType name="dam_v001_mathitemType">
        ...
        """
        blanks = ' ' * (indent*4)
        subs = []
        for modelname in cls.get_modelnames():
            model = importtools.prepare_model(modelname)
            subs.extend([
                f'{blanks}<complexType name="{modelname}_mathitemType">',
                f'{blanks}    <complexContent>',
                f'{blanks}        <extension base="hpcb:setitemType">',
                f'{blanks}            <choice>'])
            for subvars in cls._get_subvars(model):
                for var in subvars:
                    subs.append(
                        f'{blanks}                '
                        f'<element name="{subvars.name}.{var.name}"/>')
            subs.extend([
                    f'{blanks}            </choice>',
                    f'{blanks}        </extension>',
                    f'{blanks}    </complexContent>',
                    f'{blanks}</complexType>',
                    f''])
        return '\n'.join(subs)