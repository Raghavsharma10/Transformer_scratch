def command_gen(self, *names):
        '''
        Runs generator functions.

        Run `docs` generator function::

            ./manage.py sqla:gen docs

        Run `docs` generator function with `count=10`::

            ./manage.py sqla:gen docs:10
        '''
        if not names:
            sys.exit('Please provide generator names')
        for name in names:
            name, count = name, 0
            if ':' in name:
                name, count = name.split(':', 1)
            count = int(count)
            create = self.generators[name]
            print('Generating `{0}` count={1}'.format(name, count))
            create(self.session, count)
            self.session.commit()