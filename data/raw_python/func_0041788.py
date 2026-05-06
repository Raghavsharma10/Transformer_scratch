def from_json(json_string):
    """
    Returns a new MessageReceived from the provided json_string string
    """
    # parse the provided json_message
    try:            
        parsed_msg = json.loads(json_string)            
    except ValueError as ex:            
        # if the provided json_message is not a valid JSON
        return None
    except TypeError as ex:
        # if json_message not string or buffer
        return None
    herald_version = None
    # check if it is a valid Herald JSON message
    if herald.MESSAGE_HEADERS in parsed_msg:
        if herald.MESSAGE_HERALD_VERSION in parsed_msg[herald.MESSAGE_HEADERS]:
            herald_version = parsed_msg[herald.MESSAGE_HEADERS].get(herald.MESSAGE_HERALD_VERSION)                         
    if herald_version is None or herald_version != herald.HERALD_SPECIFICATION_VERSION:
        _logger.error("Herald specification of the received message is not supported!")
        return None   
    # construct new Message object from the provided JSON object    
    msg = herald.beans.MessageReceived(uid=(parsed_msg[herald.MESSAGE_HEADERS].get(herald.MESSAGE_HEADER_UID) or None), 
                          subject=parsed_msg[herald.MESSAGE_SUBJECT], 
                          content=None, 
                          sender_uid=(parsed_msg[herald.MESSAGE_HEADERS].get(herald.MESSAGE_HEADER_SENDER_UID) or None), 
                          reply_to=(parsed_msg[herald.MESSAGE_HEADERS].get(herald.MESSAGE_HEADER_REPLIES_TO) or None), 
                          access=None,
                          timestamp=(parsed_msg[herald.MESSAGE_HEADERS].get(herald.MESSAGE_HEADER_TIMESTAMP) or None) 
                          )                           
    # set content
    try:
        if herald.MESSAGE_CONTENT in parsed_msg:
            parsed_content = parsed_msg[herald.MESSAGE_CONTENT]                              
            if parsed_content is not None:
                if isinstance(parsed_content, str):
                    msg.set_content(parsed_content)
                else:
                    msg.set_content(jabsorb.from_jabsorb(parsed_content))
    except KeyError as ex:
        _logger.error("Error retrieving message content! " + str(ex)) 
    # other headers
    if herald.MESSAGE_HEADERS in parsed_msg:
        for key in parsed_msg[herald.MESSAGE_HEADERS]:
            if key not in msg._headers:
                msg._headers[key] = parsed_msg[herald.MESSAGE_HEADERS][key]         
    # metadata
    if herald.MESSAGE_METADATA in parsed_msg:
        for key in parsed_msg[herald.MESSAGE_METADATA]:
            if key not in msg._metadata:
                msg._metadata[key] = parsed_msg[herald.MESSAGE_METADATA][key] 
                       
    return msg