def _set_var_from_xml_text(self, xml, xmlpath, var):
        '''
        Sets a object variable from the xml if it is there
        and passing it through a data conversion based on the variable datatype
        '''
        xmle = xml.find(xmlpath)
        if xmle is not None:
            setattr(self, var, type_converter[ xmle.attrib.get('datatype', 'string') ]( xmle.text ))