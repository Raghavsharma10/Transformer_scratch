def perform_checks(template,
                   do_redirect=False,
                   context=None,
                   next=None,
                   quiet=False):

    '''return all checks for required variables before returning to 
       desired view

       Parameters
       ==========
       template: the html template to render
       do_redirect: if True, perform a redirect and not render
       context: dictionary of context variables to pass to render_template
       next: a pre-defined next experiment, will calculate if None
       quiet: decrease verbosity

    '''
    from expfactory.server import app
    username = session.get('username')
    subid = session.get('subid')

    # If redirect, "last" is currently active (about to start)
    # If render, "last" is last completed / active experiment (just finished)
    last = session.get('exp_id')
    if next is None:
        next = app.get_next(session)
    session['exp_id'] = next

    # Headless mode requires token
    if "token" not in session and app.headless is True:
        flash('A token is required for these experiments.')
        return redirect('/')

    # Update the user / log
    if quiet is False:
        app.logger.info("[router] %s --> %s [subid] %s [user] %s" %(last,
                                                                    next, 
                                                                    subid,
                                                                    username))

    if username is None and app.headless is False:
        flash('You must start a session before doing experiments.')
        return redirect('/')

    if subid is None:
        flash('You must have a participant identifier before doing experiments')
        return redirect('/')

    if next is None:
        flash('Congratulations, you have finished the battery!')
        return redirect('/finish')

    if do_redirect is True:
        app.logger.debug('Redirecting to %s' %template)
        return redirect(template)

    if context is not None and isinstance(context, dict):
        app.logger.debug('Rendering %s' %template)
        return render_template(template, **context)
    return render_template(template)