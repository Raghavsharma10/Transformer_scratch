def zset_example():
    """ Example sorted set pagination.
    """
    from uuid import uuid4
    from redis import StrictRedis
    from zato.redis_paginator import ZSetPaginator
    
    conn = StrictRedis()
    key = 'paginator:{}'.format(uuid4().hex)
    
    # 97-114 is 'a' to 'r' in ASCII
    for x in range(1, 18):
        conn.zadd(key, x, chr(96 + x))
        
    p = ZSetPaginator(conn, key, 6)
    
    print(p.count)      # 17
    print(p.num_pages)  # 3
    print(p.page_range) # [1, 2, 3]
    
    page = p.page(3)
    print(page)             # <Page 3 of 3>
    print(page.object_list) # ['m', 'n', 'o', 'p', 'q']
        
    conn.delete(key)