def render(self, context):
    '''Evaluates the code in the page and returns the result'''
    modules = {
    }
    context['false'] = False
    context['true'] = True
    new_ctx = eval('dict(%s)'%self.code,modules,context)
    context.update(new_ctx)
    return ''