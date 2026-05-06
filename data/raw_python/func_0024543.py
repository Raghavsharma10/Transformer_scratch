def sync_wf_cache(current):
    """
    BG Job for storing wf state to DB
    """
    wf_cache = WFCache(current)
    wf_state = wf_cache.get()  # unicode serialized json to dict, all values are unicode
    if 'role_id' in wf_state:
        # role_id inserted by engine, so it's a sign that we get it from cache not db
        try:
            wfi = WFInstance.objects.get(key=current.input['token'])
        except ObjectDoesNotExist:
            # wf's that not started from a task invitation
            wfi = WFInstance(key=current.input['token'])
            wfi.wf = BPMNWorkflow.objects.get(name=wf_state['name'])
        if not wfi.current_actor.exist:
            # we just started the wf
            try:
                inv = TaskInvitation.objects.get(instance=wfi, role_id=wf_state['role_id'])
                inv.delete_other_invitations()
                inv.progress = 20
                inv.save()
            except ObjectDoesNotExist:
                current.log.exception("Invitation not found: %s" % wf_state)
            except MultipleObjectsReturned:
                current.log.exception("Multiple invitations found: %s" % wf_state)
        wfi.step = wf_state['step']
        wfi.name = wf_state['name']
        wfi.pool = wf_state['pool']
        wfi.current_actor_id = str(wf_state['role_id'])  # keys must be str not unicode
        wfi.data = wf_state['data']
        if wf_state['finished']:
            wfi.finished = True
            wfi.finish_date = wf_state['finish_date']
            wf_cache.delete()
        wfi.save()

    else:
        # if cache already cleared, we have nothing to sync
        pass