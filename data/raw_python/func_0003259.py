def syscall_removequeue(queue, index):
    '''
    Remove subqueue `queue[index]` from queue.
    '''
    def _syscall(scheduler, processor):
        events = scheduler.queue.unblockqueue(queue[index])
        for e in events:
            scheduler.eventtree.remove(e)
        qes, qees = queue.removeSubQueue(index)
        for e in qes:
            processor(e)
        for e in qees:
            processor(e)
    return _syscall