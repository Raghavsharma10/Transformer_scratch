def init_pkg(pkg, repo_dest):
    """
    Initializes a custom named package module.

    This works by replacing all instances of 'project' with a custom module
    name.
    """
    vars = {'pkg': pkg}
    with dir_path(repo_dest):
        patch("""\
        diff --git a/manage.py b/manage.py
        index 40ebb0a..cdfe363 100755
        --- a/manage.py
        +++ b/manage.py
        @@ -3,7 +3,7 @@ import os
         import sys

         # Edit this if necessary or override the variable in your environment.
        -os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
        +os.environ.setdefault('DJANGO_SETTINGS_MODULE', '%(pkg)s.settings')

         try:
             # For local development in a virtualenv:
        diff --git a/project/settings/base.py b/project/settings/base.py
        index 312f280..c75e673 100644
        --- a/project/settings/base.py
        +++ b/project/settings/base.py
        @@ -7,7 +7,7 @@ from funfactory.settings_base import *
         # If you did not install Playdoh with the funfactory installer script
         # you may need to edit this value. See the docs about installing from a
         # clone.
        -PROJECT_MODULE = 'project'
        +PROJECT_MODULE = '%(pkg)s'

         # Bundles is a dictionary of two dictionaries, css and js, which list css files
         # and js files that can be bundled together by the minify app.
        diff --git a/setup.py b/setup.py
        index 58dbd93..9a38628 100644
        --- a/setup.py
        +++ b/setup.py
        @@ -3,7 +3,7 @@ import os
         from setuptools import setup, find_packages


        -setup(name='project',
        +setup(name='%(pkg)s',
               version='1.0',
               description='Django application.',
               long_description='',
        """ % vars)

        git(['mv', 'project', pkg])
        git(['commit', '-a', '-m', 'Renamed project module to %s' % pkg])