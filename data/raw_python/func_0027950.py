def styleattribute(self, element):
        """
          returns css.CSSStyleDeclaration of inline styles, for html: @style
          """
        css_text = element.get('style')
        if css_text:
            return cssutils.css.CSSStyleDeclaration(cssText=css_text)
        else:
            return None