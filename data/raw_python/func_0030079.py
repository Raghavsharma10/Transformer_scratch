def render(self):
        '''Proxy method to form's environment render method'''
        return self.env.template.render(self.template, form=self)