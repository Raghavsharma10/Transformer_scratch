def getPublicIPaddress(timeout=c.DEFAULT_TIMEOUT):
    """visible on public internet"""
    start = time.time()
    my_public_ip = None
    e = Exception
    while my_public_ip == None:
        if time.time() - start > timeout:
            break
        try: #httpbin.org -- site is useful to test scripts / applications.
            my_public_ip = json.load(urlopen('http://httpbin.org/ip'))['origin'] # takes ~0.14s as ipv4
            if my_public_ip: break
        except Exception as e:
            print(type(e), e, "http://httpbin.org/ip")
        try: #jsonip.com -- Seemingly the sole purpose of this domain is to return IP address in JSON.
            my_public_ip = json.load(urlopen('http://jsonip.com'))['ip']  # takes ~0.24s as ipv6
            if my_public_ip: break
        except Exception as e:
            print(type(e), e, "http://jsonip.com")
        try: #ipify.org -- Power of this service results from lack of limits (there is no rate limiting), infrastructure (placed on Heroku, with high availability in mind) and flexibility (works for both IPv4 and IPv6).
            my_public_ip = load(urlopen('https://api.ipify.org/?format=json'))['ip']  # takes ~0.33s
            if my_public_ip: break
        except Exception as e:
            print(type(e), e, "https://api.ipify.org/")
        try: #ip.42.pl -- This is very convenient for scripts, you don't need JSON parsing here.
            my_public_ip = urlopen('http://ip.42.pl/raw').read()  # takes ~0.35s
            if my_public_ip: break
        except Exception as e:
            print(type(e), e, "http://ip.42.pl/raw")
    if not my_public_ip:
        raise e
    return my_public_ip