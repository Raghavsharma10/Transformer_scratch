def rate_limit_info():
    """ Returns (requests_remaining, minutes_to_reset) """
    import json
    import time

    r = requests.get(gh_url + "/rate_limit", auth=login.auth())
    out = json.loads(r.text)
    mins = (out["resources"]["core"]["reset"]-time.time())/60
    return out["resources"]["core"]["remaining"], mins