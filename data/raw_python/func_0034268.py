def init( self, lang='en', css=None, metainfo=None, title=None, header=None,
              footer=None, charset=None, encoding=None, doctype=None, bodyattrs=None, script=None, base=None ):
        """This method is used for complete documents with appropriate
        doctype, encoding, title, etc information. For an HTML/XML snippet
        omit this method.

        lang --     language, usually a two character string, will appear
                    as <html lang='en'> in html mode (ignored in xml mode)
        
        css --      Cascading Style Sheet filename as a string or a list of
                    strings for multiple css files (ignored in xml mode)

        metainfo -- a dictionary in the form { 'name':'content' } to be inserted
                    into meta element(s) as <meta name='name' content='content'>
                    (ignored in xml mode)
        
        base     -- set the <base href="..."> tag in <head>
        
        bodyattrs --a dictionary in the form { 'key':'value', ... } which will be added
                    as attributes of the <body> element as <body key='value' ... >
                    (ignored in xml mode)

        script --   dictionary containing src:type pairs, <script type='text/type' src=src></script>
                    or a list of [ 'src1', 'src2', ... ] in which case 'javascript' is assumed for all

        title --    the title of the document as a string to be inserted into
                    a title element as <title>my title</title> (ignored in xml mode)

        header --   some text to be inserted right after the <body> element
                    (ignored in xml mode)

        footer --   some text to be inserted right before the </body> element
                    (ignored in xml mode)

        charset --  a string defining the character set, will be inserted into a
                    <meta http-equiv='Content-Type' content='text/html; charset=myset'>
                    element (ignored in xml mode)

        encoding -- a string defining the encoding, will be put into to first line of
                    the document as <?xml version='1.0' encoding='myencoding' ?> in
                    xml mode (ignored in html mode)

        doctype --  the document type string, defaults to
                    <!DOCTYPE HTML PUBLIC '-//W3C//DTD HTML 4.01 Transitional//EN'>
                    in html mode (ignored in xml mode)"""

        self._full = True

        if self.mode == 'strict_html' or self.mode == 'loose_html':
            if doctype is None:
                doctype = "<!DOCTYPE HTML PUBLIC '-//W3C//DTD HTML 4.01 Transitional//EN'>"
            self.header.append( doctype )
            self.html( lang=lang )
            self.head( )
            if charset is not None:
                self.meta( http_equiv='Content-Type', content="text/html; charset=%s" % charset )
            if metainfo is not None:
                self.metainfo( metainfo )
            if css is not None:
                self.css( css )
            if title is not None:
                self.title( title )
            if script is not None:
                self.scripts( script )
            if base is not None:
                self.base( href='%s' % base )
            self.head.close()
            if bodyattrs is not None:
                self.body( **bodyattrs )
            else:
                self.body( )
            if header is not None:
                self.content.append( header )
            if footer is not None:
                self.footer.append( footer )

        elif self.mode == 'xml':
            if doctype is None:
                if encoding is not None:
                    doctype = "<?xml version='1.0' encoding='%s' ?>" % encoding
                else:
                    doctype = "<?xml version='1.0' ?>"
            self.header.append( doctype )