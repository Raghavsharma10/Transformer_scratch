def format_output(self, rendered_widgets):
        """
        This output will yeild all widgets grouped in a un-ordered list
        """
        ret = [u'<ul class="formfield">']
        for i, field in enumerate(self.fields):
            label = self.format_label(field, i)
            help_text = self.format_help_text(field, i)
            ret.append(u'<li>%s %s %s</li>' % (
                label, rendered_widgets[i], field.help_text and help_text))

        ret.append(u'</ul>')
        return ''.join(ret)