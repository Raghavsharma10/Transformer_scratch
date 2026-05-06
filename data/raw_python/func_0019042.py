def get_subgroupiteminsertion(
            cls, itemgroup, model, subgroup, indent) -> str:
        """Return a string defining the required types for the given
        combination of an exchange item group and a specific variable
        subgroup of an application model or class |Node|.

        Note that for `setitems` and `getitems` `setitemType` and
        `getitemType` are referenced, respectively, and for all others
        the model specific `mathitemType`:

        >>> from hydpy import prepare_model
        >>> model = prepare_model('hland_v1')
        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_subgroupiteminsertion(    # doctest: +ELLIPSIS
        ...     'setitems', model, model.parameters.control, 1))
            <element name="control"
                     minOccurs="0"
                     maxOccurs="unbounded">
                <complexType>
                    <sequence>
                        <element ref="hpcb:selections"
                                 minOccurs="0"/>
                        <element ref="hpcb:devices"
                                 minOccurs="0"/>
                        <element name="area"
                                 type="hpcb:setitemType"
                                 minOccurs="0"
                                 maxOccurs="unbounded"/>
                        <element name="nmbzones"
        ...
                    </sequence>
                </complexType>
            </element>

        >>> print(XSDWriter.get_subgroupiteminsertion(    # doctest: +ELLIPSIS
        ...     'getitems', model, model.parameters.control, 1))
            <element name="control"
        ...
                        <element name="area"
                                 type="hpcb:getitemType"
                                 minOccurs="0"
                                 maxOccurs="unbounded"/>
        ...

        >>> print(XSDWriter.get_subgroupiteminsertion(    # doctest: +ELLIPSIS
        ...     'additems', model, model.parameters.control, 1))
            <element name="control"
        ...
                        <element name="area"
                                 type="hpcb:hland_v1_mathitemType"
                                 minOccurs="0"
                                 maxOccurs="unbounded"/>
        ...

        For sequence classes, additional "series" elements are added:

        >>> print(XSDWriter.get_subgroupiteminsertion(    # doctest: +ELLIPSIS
        ...     'setitems', model, model.sequences.fluxes, 1))
            <element name="fluxes"
        ...
                        <element name="tmean"
                                 type="hpcb:setitemType"
                                 minOccurs="0"
                                 maxOccurs="unbounded"/>
                        <element name="tmean.series"
                                 type="hpcb:setitemType"
                                 minOccurs="0"
                                 maxOccurs="unbounded"/>
                        <element name="tc"
        ...
                    </sequence>
                </complexType>
            </element>
        """
        blanks1 = ' ' * (indent * 4)
        blanks2 = ' ' * ((indent+5) * 4 + 1)
        subs = [
            f'{blanks1}<element name="{subgroup.name}"',
            f'{blanks1}         minOccurs="0"',
            f'{blanks1}         maxOccurs="unbounded">',
            f'{blanks1}    <complexType>',
            f'{blanks1}        <sequence>',
            f'{blanks1}            <element ref="hpcb:selections"',
            f'{blanks1}                     minOccurs="0"/>',
            f'{blanks1}            <element ref="hpcb:devices"',
            f'{blanks1}                     minOccurs="0"/>']
        seriesflags = (False,) if subgroup.name == 'control' else (False, True)
        for variable in subgroup:
            for series in seriesflags:
                name = f'{variable.name}.series' if series else variable.name
                subs.append(f'{blanks1}            <element name="{name}"')
                if itemgroup == 'setitems':
                    subs.append(f'{blanks2}type="hpcb:setitemType"')
                elif itemgroup == 'getitems':
                    subs.append(f'{blanks2}type="hpcb:getitemType"')
                else:
                    subs.append(
                        f'{blanks2}type="hpcb:{model.name}_mathitemType"')
                subs.append(f'{blanks2}minOccurs="0"')
                subs.append(f'{blanks2}maxOccurs="unbounded"/>')
        subs.extend([
            f'{blanks1}        </sequence>',
            f'{blanks1}    </complexType>',
            f'{blanks1}</element>'])
        return '\n'.join(subs)