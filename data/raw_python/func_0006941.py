def install_oracle_java(distribution, java_version):
    """ installs oracle java """
    if 'ubuntu' in distribution:
        accept_oracle_license = ('echo '
                                 'oracle-java' + java_version + 'installer '
                                 'shared/accepted-oracle-license-v1-1 '
                                 'select true | '
                                 '/usr/bin/debconf-set-selections')
        with settings(hide('running', 'stdout')):
            sudo(accept_oracle_license)

        with settings(hide('running', 'stdout'),
                      prompts={"Press [ENTER] to continue or ctrl-c to cancel adding it": "yes"}): # noqa
            sudo("yes | add-apt-repository ppa:webupd8team/java")

        with settings(hide('running', 'stdout')):
            sudo('DEBIAN_FRONTEND=noninteractive apt-get update')
            apt_install(packages=['oracle-java8-installer',
                                  'oracle-java8-set-default'])