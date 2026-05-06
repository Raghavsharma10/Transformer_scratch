def to_json(msg):
    """
    Returns a JSON string representation of this message
    """
    result = {}
    
    # herald specification version
    #result[herald.MESSAGE_HERALD_VERSION] = herald.HERALD_SPECIFICATION_VERSION
    
    # headers
    result[herald.MESSAGE_HEADERS] = {}        
    if msg.headers is not None:
        for key in msg.headers:
            result[herald.MESSAGE_HEADERS][key] = msg.headers.get(key) or None        
    
    # subject
    result[herald.MESSAGE_SUBJECT] = msg.subject
    # content
    if msg.content is not None:
        if isinstance(msg.content, str):
            # string content
            result[herald.MESSAGE_CONTENT] = msg.content
        else:
            # jaborb content
            result[herald.MESSAGE_CONTENT] = jabsorb.to_jabsorb(msg.content)
    
    # metadata
    result[herald.MESSAGE_METADATA] = {}        
    if msg.metadata is not None:
        for key in msg.metadata:
            result[herald.MESSAGE_METADATA][key] = msg.metadata.get(key) or None
            
    return json.dumps(result, default=herald.utils.json_converter)