def getUserInfoError(sAccessToken):
    """
        May be return {u'msg': u'Access_token repealed', u'errno': u'-102', u'data': []}
    """
    import urllib.request, urllib.parse, urllib.error
    payload = urllib.parse.urlencode({'access_token': sAccessToken})
    c = http.client.HTTPSConnection('796.com')
    c.request("GET", "/v1/user/get_info?"+payload)
    r = c.getresponse()
    data = r.read()
    jsonDict = json.loads(data.decode('utf-8'));
    print(jsonDict)