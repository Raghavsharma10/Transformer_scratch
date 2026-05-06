def _set_list_ids_from_xml_iter(self, xml, xmlpath, var):
        '''
        Set a list variable from the frameids of matching xml entities
        '''
        es = xml.iterfind(xmlpath)
        if es is not None:
            l = []
            for e in es:
                l.append( e.attrib['frameid'] )
            
            setattr(self, var, l)