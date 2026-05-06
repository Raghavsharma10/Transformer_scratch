def get_response(cmd, conn):
    """Return a response"""
    resp = conn.socket().makefile('rb', -1)
    resp_dict = dict(
        code=0,
        message='',
        isspam=False,
        score=0.0,
        basescore=0.0,
        report=[],
        symbols=[],
        headers={},
    )

    if cmd == 'TELL':
        resp_dict['didset'] = False
        resp_dict['didremove'] = False

    data = resp.read()
    lines = data.split('\r\n')
    for index, line in enumerate(lines):
        if index == 0:
            match = RESPONSE_RE.match(line)
            if not match:
                raise SpamCResponseError(
                    'spamd unrecognized response: %s' % data)
            resp_dict.update(match.groupdict())
            resp_dict['code'] = int(resp_dict['code'])
        else:
            if not line.strip():
                continue
            match = SPAM_RE.match(line)
            if match:
                tmp = match.groupdict()
                resp_dict['score'] = float(tmp['score'])
                resp_dict['basescore'] = float(tmp['basescore'])
                resp_dict['isspam'] = tmp['isspam'] in ['True', 'Yes']
            if not match:
                if cmd == 'SYMBOLS':
                    match = PART_RE.findall(line)
                    for part in match:
                        resp_dict['symbols'].append(part)
            if not match and cmd != 'PROCESS':
                match = RULE_RE.findall(line)
                if match:
                    resp_dict['report'] = []
                    for part in match:
                        score = part[0] + part[1]
                        score = score.strip()
                        resp_dict['report'].append(
                            dict(score=score,
                                 name=part[2],
                                 description=SPACE_RE.sub(" ", part[3])))
            if line.startswith('DidSet:'):
                resp_dict['didset'] = True
            if line.startswith('DidRemove:'):
                resp_dict['didremove'] = True
    if cmd == 'PROCESS':
        resp_dict['message'] = ''.join(lines[4:]) + '\r\n'
    if cmd == 'HEADERS':
        parser = Parser()
        headers = parser.parsestr('\r\n'.join(lines[4:]), headersonly=True)
        for key in headers.keys():
            resp_dict['headers'][key] = headers[key]
    return resp_dict