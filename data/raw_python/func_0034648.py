def render(self, xml, context, raise_on_errors=True):
        """Render xml string and apply XSLT transfomation with context"""

        if xml:
            self.xml = xml

            # render XSL
            self.render_xsl(self.root, context)

            # create root XSL sheet
            xsl_ns = self.namespaces['xsl']
            rootName = etree.QName(xsl_ns, 'stylesheet')
            root = etree.Element(rootName, nsmap={'xsl': xsl_ns})
            sheet = etree.ElementTree(root)
            template = etree.SubElement(root, etree.QName(xsl_ns, "template"), match='/')

            # put OpenOffice tree into XSLT sheet
            template.append(self.root)
            self.root = root

            # drop XSL styles
            self.remove_style()

            #self.debug(self.xml)

            try:
                # transform XSL
                xsl = etree.XSLT(self.root)
                self.root = xsl(context)

            except etree.Error as e:
                # log errors
                for l in e.error_log:
                    self.error("XSLT error at line %s col %s:" % (l.line, l.column))
                    self.error("    message: %s" % l.message)
                    self.error("    domain: %s (%d)" % (l.domain_name, l.domain))
                    self.error('    type: %s (%d)' % (l.type_name, l.type))
                    self.error('    level: %s (%d)' % (l.level_name, l.level))
                    self.error('    filename: %s' % l.filename)

                if raise_on_errors:
                    raise

            return self.xml

        else:
            return xml