def syscall_clearqueue(queue):
    '''
    Clear a queue
    '''
    def _syscall(scheduler, processor):
        qes, qees = queue.clear()
        events = scheduler.queue.unblockqueue(queue)
        for e in events:
            scheduler.eventtree.remove(e)
        for e in qes:
            processor(e)
        for e in qees:
            processor(e)
    return _syscall