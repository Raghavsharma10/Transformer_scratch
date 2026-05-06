def wait_and_ignore(condition, timeout=WTF_TIMEOUT_MANAGER.NORMAL, sleep=0.5):
    '''
    Waits wrapper that'll wait for the condition to become true, but will 
    not error if the condition isn't met.

    Args:
        condition (lambda) - Lambda expression to wait for to evaluate to True.

    Kwargs:
        timeout (number) : Maximum number of seconds to wait.
        sleep (number) : Sleep time to wait between iterations.

    Example::

        wait_and_ignore(lambda: driver.find_element_by_id("success").is_displayed(), 
                       timeout=30,
                       sleep=0.5)

    is equivalent to::

        end_time = datetime.now() + timedelta(seconds=30)
        while datetime.now() < end_time:
            try:
                if driver.find_element_by_id("success").is_displayed():
                    break;
            except:
                pass
            time.sleep(0.5)
    '''
    try:
        return wait_until(condition, timeout, sleep)
    except:
        pass