def subcommand(self, description='', arguments={}):
        '''
        Decorator for quickly adding subcommands to the omnic CLI
        '''
        def decorator(func):
            self.register_subparser(
                func,
                func.__name__.replace('_', '-'),
                description=description,
                arguments=arguments,
            )
            return func
        return decorator