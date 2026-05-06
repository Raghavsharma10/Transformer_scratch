def configuration():
    'Loads configuration from the file system.'
    defaults = '''
[oauth2]
hostname = localhost
port = 9876
api_endpoint = https://api.coursera.org
auth_endpoint = https://accounts.coursera.org/oauth2/v1/auth
token_endpoint = https://accounts.coursera.org/oauth2/v1/token
verify_tls = True
token_cache_base = ~/.coursera

[manage_graders]
client_id = NS8qaSX18X_Eu0pyNbLsnA
client_secret = bUqKqGywnGXEJPFrcd4Jpw
scopes = view_profile manage_graders

[manage_research_exports]
client_id = sDHC8Nfp-b1XMbzZx8Wa4w
client_secret = pgD4adDd7lm-ksfG7UazUA
scopes = view_profile manage_research_exports
'''
    cfg = ConfigParser.SafeConfigParser()
    cfg.readfp(io.BytesIO(defaults))
    cfg.read([
        '/etc/coursera/courseraoauth2client.cfg',
        os.path.expanduser('~/.coursera/courseraoauth2client.cfg'),
        'courseraoauth2client.cfg',
    ])
    return cfg