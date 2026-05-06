def main():
    """Main"""
    lic = (
        'License :: OSI Approved :: GNU Affero '
        'General Public License v3 or later (AGPLv3+)')
    version = load_source("version", os.path.join("spamc", "version.py"))

    opts = dict(
        name="spamc",
        version=version.__version__,
        description="Python spamassassin spamc client library",
        long_description=get_readme(),
        keywords="spam spamc spamassassin",
        author="Andrew Colin Kissa",
        author_email="andrew@topdog.za.net",
        url="https://github.com/akissa/spamc",
        license="AGPLv3+",
        packages=find_packages(exclude=['tests']),
        include_package_data=True,
        zip_safe=False,
        tests_require=TESTS_REQUIRE,
        install_requires=INSTALL_REQUIRES,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Intended Audience :: Developers',
            lic,
            'Natural Language :: English',
            'Operating System :: OS Independent'],)
    setup(**opts)