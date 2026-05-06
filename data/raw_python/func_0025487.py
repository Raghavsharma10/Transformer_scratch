def documento(self, *args, **kwargs):
        """Resulta no documento XML como string, que pode ou não incluir a
        declaração XML no início do documento.
        """
        forcar_unicode = kwargs.pop('forcar_unicode', False)
        incluir_xml_decl = kwargs.pop('incluir_xml_decl', True)
        doc = ET.tostring(self._xml(*args, **kwargs),
                encoding='utf-8').decode('utf-8')
        if forcar_unicode:
            if incluir_xml_decl:
                doc = u'{}\n{}'.format(constantes.XML_DECL_UNICODE, doc)
        else:
            if incluir_xml_decl:
                doc = '{}\n{}'.format(constantes.XML_DECL, unidecode(doc))
            else:
                doc = unidecode(doc)
        return doc