def list_example():
    """ Example list pagination.
    """
    from uuid import uuid4
    from redis import StrictRedis
    from zato.redis_paginator import ListPaginator
    
    conn = StrictRedis()
    key = 'paginator:{}'.format(uuid4().hex)
    
    for x in range(1, 18):
        conn.rpush(key, x)
        
    p = ListPaginator(conn, key, 6)
    
    print(p.count)      # 17
    print(p.num_pages)  # 3
    print(p.page_range) # [1, 2, 3]
    
    page = p.page(3)
    print(page)             # <Page 3 of 3>
    print(page.object_list) # ['13', '14', '15', '16', '17']
        
    conn.delete(key)