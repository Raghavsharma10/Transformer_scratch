def build_final_response(request, meta, result, menu, hproject, proxyMode, context):
    """Build the final response to send back to the browser"""

    if 'no_template' in meta and meta['no_template']:  # Just send the json back
        return HttpResponse(result)

    # TODO this breaks pages not using new template
    # Add sidebar toggler if plugit did not add by itself
    # if not "sidebar-toggler" in result:
    #     result = "<div class=\"menubar\"><div class=\"sidebar-toggler visible-xs\"><i class=\"ion-navicon\"></i></div></div>" + result

    # render the template into the whole page
    if not settings.PIAPI_STANDALONE:
        return render_to_response('plugIt/' + hproject.get_plugItTemplate_display(),
                                  {"project": hproject,
                                   "plugit_content": result,
                                   "plugit_menu": menu,
                                   'context': context},
                                  context_instance=RequestContext(request))

    if proxyMode:  # Force inclusion inside template
        return render_to_response('plugIt/base.html',
                                  {'plugit_content': result,
                                   "plugit_menu": menu,
                                   'context': context},
                                  context_instance=RequestContext(request))

    renderPlugItTemplate = 'plugItBase.html'
    if settings.PIAPI_PLUGITTEMPLATE:
        renderPlugItTemplate = settings.PIAPI_PLUGITTEMPLATE

    return render_to_response('plugIt/' + renderPlugItTemplate,
                              {"plugit_content": result,
                               "plugit_menu": menu,
                               'context': context},
                              context_instance=RequestContext(request))