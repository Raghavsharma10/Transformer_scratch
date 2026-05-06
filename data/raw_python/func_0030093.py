def render(self, template_name, **kw):
        'Interface method called from `Template.render`'
        return self.env.get_template(template_name).render(**kw)