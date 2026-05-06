def extra(request, provider):
    """
        Handle registration of new user with extra data for profile
    """
    identity = request.session.get('identity', None)
    if not identity:
        raise Http404

    if request.method == "POST":
        form = str_to_class(settings.EXTRA_FORM)(request.POST)
        if form.is_valid():
            user = form.save(request, identity, provider)
            del request.session['identity']
            if not settings.ACTIVATION_REQUIRED:
                user = auth.authenticate(identity=identity, provider=provider)
                if user:
                    auth.login(request, user)
                    return redirect(request.session.pop('next_url', settings.LOGIN_REDIRECT_URL))
            else:
                messages.warning(request, lang.ACTIVATION_REQUIRED_TEXT)
                return redirect(settings.ACTIVATION_REDIRECT_URL)
    else:
        initial = request.session['extra']
        form = str_to_class(settings.EXTRA_FORM)(initial=initial)

    return render_to_response('netauth/extra.html', {'form': form }, context_instance=RequestContext(request))