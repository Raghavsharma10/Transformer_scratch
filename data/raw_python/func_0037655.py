def main():

    ''' set things up '''
    configs =  setup(argparse.ArgumentParser())
    harvester = GreyHarvester(
        test_domain=configs['test_domain'],
        test_sleeptime=TEST_SLEEPTIME,
        https_only=configs['https_only'],
        allowed_countries=configs['allowed_countries'],
        denied_countries=configs['denied_countries'],
        ports=configs['ports'],
        max_timeout=configs['max_timeout']
    )

    ''' harvest free and working proxies from teh interwebz '''
    count = 0
    for proxy in harvester.run():
        if count >= configs['num_proxies']:
            break
        print(proxy)
        count += 1