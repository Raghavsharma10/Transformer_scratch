def xml_row(row, lang):
    '''
    Generator for an XML row
    '''
    for elem in row:
        name = elem.get('name')
        child = elem[0]
        ftype = re.sub(r'\{[^}]+\}', '', child.tag)
        if ftype == 'literal':
            ftype = '{}, {}'.format(ftype, child.attrib.get(XML_LANG, 'none'))
        yield (name, (child.text, ftype))