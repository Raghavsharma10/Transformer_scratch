def respond(template, context={}, request=None, **kwargs):
    "Calls render_to_response with a RequestConext"
    from django.http import HttpResponse
    from django.template import RequestContext
    from django.template.loader import render_to_string    

    if request:
        default = context_processors.default(request)
        default.update(context)
    else:
        default = context.copy()
    
    rendered = render_to_string(template, default, context_instance=request and RequestContext(request) or None)
    return HttpResponse(rendered, **kwargs)