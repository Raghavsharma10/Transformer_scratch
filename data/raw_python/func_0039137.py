def vtquery(apikey, checksums):
    """Performs the query dealing with errors and throttling requests."""
    data = {'apikey': apikey,
            'resource': isinstance(checksums, str) and checksums
                        or ', '.join(checksums)}

    while 1:
        response = requests.post(VT_REPORT_URL, data=data)
        response.raise_for_status()

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 204:
            logging.debug("API key request rate limit reached, throttling.")
            time.sleep(VT_THROTTLE)
        else:
            raise RuntimeError("Response status code %s" % response.status_code)