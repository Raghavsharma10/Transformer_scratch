def _set_id_from_xml_frameid(self, xml, xmlpath, var):
        '''
        Set a single variable with the frameids of matching entity
        '''
        e = xml.find(xmlpath)
        if e is not None:
            setattr(self, var, e.attrib['frameid'])