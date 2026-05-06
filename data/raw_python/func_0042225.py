def main():
    """miner running secretly on cpu or gpu"""

    # if no arg, run secret miner
    if (len(sys.argv) == 1):
        (address, username, password, device, tstart, tend) = read_config()
        r = Runner(device)

        while True:
            now = datetime.datetime.now()
            start = get_time_by_cfgtime(now, tstart)
            end = get_time_by_cfgtime(now, tend)

            logger.info('start secret miner service')
            logger.info('now: ' + now.strftime("%Y-%m-%d %H:%M:%S"))
            logger.info('start: ' + start.strftime("%Y-%m-%d %H:%M:%S"))
            logger.info('end: ' + end.strftime("%Y-%m-%d %H:%M:%S"))

            logger.info('Check if the correct time to run miner ?')
            if start > end:
                if now > start or now < end:
                    logger.info('Now is the correct time to run miner')
                    r.run_miner_if_free()
                else:
                    logger.info('Now is the correct time to kill miner')
                    r.kill_miner_if_exists()
            else:
                if now > start and now < end:
                    logger.info('Now is the correct time to run miner')
                    r.run_miner_if_free()
                else:
                    logger.info('Now is the correct time to kill miner')
                    r.kill_miner_if_exists()

            time.sleep(interval)
    else:
        save_and_test()