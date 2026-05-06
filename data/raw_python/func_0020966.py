def unpack_frame(message):
    """Called to unpack a STOMP message into a dictionary.

    returned = {
        # STOMP Command:
        'cmd' : '...',

        # Headers e.g.
        'headers' : {
            'destination' : 'xyz',
            'message-id' : 'some event',
            :
            etc,
        }

        # Body:
        'body' : '...1234...\x00',
    }

    """
    body = []
    returned = dict(cmd='', headers={}, body='')

    breakdown = message.split('\n')

    # Get the message command:
    returned['cmd'] = breakdown[0]
    breakdown = breakdown[1:]

    def headD(field):
        # find the first ':' everything to the left of this is a
        # header, everything to the right is data:
        index = field.find(':')
        if index:
            header = field[:index].strip()
            data = field[index+1:].strip()
#            print "header '%s' data '%s'" % (header, data)
            returned['headers'][header.strip()] = data.strip()

    def bodyD(field):
        field = field.strip()
        if field:
            body.append(field)

    # Recover the header fields and body data
    handler = headD
    for field in breakdown:
#        print "field:", field
        if field.strip() == '':
            # End of headers, it body data next.
            handler = bodyD
            continue

        handler(field)

    # Stich the body data together:
#    print "1. body: ", body
    body = "".join(body)
    returned['body'] = body.replace('\x00', '')

#    print "2. body: <%s>" % returned['body']

    return returned