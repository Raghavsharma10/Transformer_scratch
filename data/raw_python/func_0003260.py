def syscall_clearremovequeue(queue, index):
    '''
    Clear the subqueue `queue[index]` and remove it from queue.
    '''
    def _syscall(scheduler, processor):
        qes, qees = queue[index].clear()
        events = scheduler.queue.unblockqueue(queue[index])
        for e in events:
            scheduler.eventtree.remove(e)
        qes2, qees2 = queue.removeSubQueue(index)
        for e in qes:
            processor(e)
        for e in qes2:
            processor(e)
        for e in qees:
            processor(e)
        for e in qees2:
            processor(e)
    return _syscall