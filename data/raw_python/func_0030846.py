def render(self):
        '''
        Renders widget to template
        '''
        data = self.prepare_data()
        if self.field.readable:
            return self.env.template.render(self.template, **data)
        return ''