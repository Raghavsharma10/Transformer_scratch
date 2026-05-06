def to_html(self, codebase):
        """
        Convert this to HTML.
        """
        html = ''
        def build_line(key, include_pred, format_fn):
            val = getattr(self, key)
            if include_pred(val):
                return '<dt>%s</dt><dd>%s</dd>\n' % (printable(key), format_fn(val))
            else:
                return ''
        def build_dependency(val):
            return ', '.join('<a href = "%s.html">%s</a>' % (trim_js_ext(name), name)
                             for name in val)
        for key in ('author', 'organization', 'version', 'license'):
            html += build_line(key, lambda val: val, lambda val: val)
        html += build_line('dependencies', lambda val: val, build_dependency)
        html += build_line('all_dependencies', lambda val: len(val) > 1, 
                                                build_dependency)
        html += codebase.build_see_html(self.see, 'h3')
        
        if html:
            return '<dl class = "module">\n%s\n</dl>\n' % html
        else:
            return ''