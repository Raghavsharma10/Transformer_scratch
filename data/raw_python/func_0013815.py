def render(self, context):
    '''Evaluates the code in the page and returns the result'''
    modules = {
      'pyjade': __import__('pyjade')
    }
    context['false'] = False
    context['true'] = True
    try:
        return six.text_type(eval('pyjade.runtime.attrs(%s)'%self.code,modules,context))
    except NameError:
        return ''