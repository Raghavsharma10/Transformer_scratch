def until(method, timeout = 30, message=''):
        """Calls the method until the return value is not False."""
        end_time = time.time() + timeout
        while True:
            try:
                value = method()
                if value:
                    return value
            except:
                pass            
            time.sleep(1)
            if time.time() > end_time:
                break
        raise Exception(message)