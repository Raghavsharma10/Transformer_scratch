def save():
    '''save is a view to save data. We might want to adjust this to allow for
       updating saved data, but given single file is just one post for now
    '''
    if request.method == 'POST':
        exp_id = session.get('exp_id')
        app.logger.debug('Saving data for %s' %exp_id)

        fields = get_post_fields(request)
        result_file = app.save_data(session=session, content=fields, exp_id=exp_id)

        experiments = app.finish_experiment(session, exp_id)
        app.logger.info('Finished %s, %s remaining.' % (exp_id, len(experiments)))

        # Note, this doesn't seem to be enough to trigger ajax success
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
    return json.dumps({'success':False}), 403, {'ContentType':'application/json'}