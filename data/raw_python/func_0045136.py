def heapremove(heap,item):
    """
    Removes item from heap.
    (This function is missing from the standard heapq package.)
    """
    i=heap.index(item)
    lastelt=heap.pop()
    if item==lastelt:
        return
    heap[i]=lastelt
    heapq._siftup(heap,i)
    if i:
        heapq._siftdown(heap,0,i)