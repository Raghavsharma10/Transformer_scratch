def make_document(self, titlestring):
        """
        This method may be used to create a new document for writing as xml
        to the OPS subdirectory of the ePub structure.
        """
        #root = etree.XML('''<?xml version="1.0"?>\
#<!DOCTYPE html  PUBLIC '-//W3C//DTD XHTML 1.1//EN'  'http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd'>\
#<html xml:lang="en-US" xmlns="http://www.w3.org/1999/xhtml" xmlns:ops="http://www.idpf.org/2007/ops">\
#</html>''')

        root = etree.XML('''<?xml version="1.0"?>\
<!DOCTYPE html>\
<html xmlns="http://www.w3.org/1999/xhtml">\
</html>''')

        document = etree.ElementTree(root)
        html = document.getroot()
        head = etree.SubElement(html, 'head')
        etree.SubElement(html, 'body')
        title = etree.SubElement(head, 'title')
        title.text = titlestring
        #The href for the css stylesheet is a standin, can be overwritten
        etree.SubElement(head,
                         'link',
                         {'href': 'css/default.css',
                          'rel': 'stylesheet',
                          'type': 'text/css'})

        return document