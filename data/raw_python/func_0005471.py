def main():
    """Sample usage for this python module

    This main method simply illustrates sample usage for this python
    module.

    :return: None
    """
    mkdir_p('/tmp/test/test')
    source('/root/.bash_profile')
    yum_install(['httpd', 'git'])
    yum_install(['httpd', 'git'], dest_dir='/tmp/test/test', downloadonly=True)
    sed('/Users/yennaco/Downloads/homer_testing/network', '^HOSTNAME.*', 'HOSTNAME=foo.joe')
    test_script = '/Users/yennaco/Downloads/homer/script.sh'
    results = run_command([test_script], timeout_sec=1000)
    print('Script {s} produced exit code [{c}] and output:\n{o}'.format(
        s=test_script, c=results['code'], o=results['output']))