def convert_disp_quote_elements(self):
        """
        Extract or extended quoted passage from another work, usually made
        typographically distinct from surrounding text

        <disp-quote> elements have a relatively complex content model, but PLoS
        appears to employ either <p>s or <list>s.
        """
        for disp_quote in self.main.getroot().findall('.//disp-quote'):
            if disp_quote.getparent().tag == 'p':
                elevate_element(disp_quote)
            disp_quote.tag = 'div'
            disp_quote.attrib['class'] = 'disp-quote'