def main():
    """The main entry point."""
    if sys.version_info < (2, 7):
        sys.exit('crispy requires at least Python 2.7')
    elif sys.version_info[0] == 3 and sys.version_info < (3, 4):
        sys.exit('crispy requires at least Python 3.4')

    kwargs = dict(
        name='crispy',
        version=get_version(),
        description='Core-Level Spectroscopy Simulations in Python',
        long_description=get_readme(),
        license='MIT',
        author='Marius Retegan',
        author_email='marius.retegan@esrf.eu',
        url='https://github.com/mretegan/crispy',
        download_url='https://github.com/mretegan/crispy/releases',
        keywords='gui, spectroscopy, simulation, synchrotron, science',
        install_requires=get_requirements(),
        platforms=[
            'MacOS :: MacOS X',
            'Microsoft :: Windows',
            'POSIX :: Linux',
        ],
        packages=[
            'crispy',
            'crispy.gui',
            'crispy.gui.uis',
            'crispy.gui.icons',
            'crispy.modules',
            'crispy.modules.quanty',
            'crispy.modules.orca',
            'crispy.utils',
        ],
        package_data={
            'crispy.gui.uis': [
                '*.ui',
                'quanty/*.ui',
            ],
            'crispy.gui.icons': [
                '*.svg',
            ],
            'crispy.modules.quanty': [
                'parameters/*.json.gz',
                'templates/*.lua',
            ],
        },
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: X11 Applications :: Qt',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering :: Visualization',
        ]
    )

    # At the moment pip/setuptools doesn't play nice with shebang paths
    # containing white spaces.
    # See: https://github.com/pypa/pip/issues/2783
    #      https://github.com/xonsh/xonsh/issues/879
    # The most straight forward workaround is to have a .bat script to run
    # crispy on Windows.

    if 'win32' in sys.platform:
        kwargs['scripts'] = ['scripts/crispy.bat']
    else:
        kwargs['scripts'] = ['scripts/crispy']

    setup(**kwargs)