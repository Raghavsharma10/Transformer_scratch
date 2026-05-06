def render(self, template_name, **kw):
        '''
        Given a template name and template vars.
        Searches a template file based on engine set, and renders it 
        with corresponding engine.
        Returns a string.
        '''
        logger.debug('Rendering template "%s"', template_name)
        vars = self.globs.copy()
        vars.update(kw)
        resolved_name, engine = self.resolve(template_name)
        return engine.render(resolved_name, **vars)