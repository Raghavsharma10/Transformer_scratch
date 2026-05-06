def help(self):
        """Prints discovered resources and their associated methods. Nice when
        noodling in the terminal to wrap your head around Magento's insanity.
        """

        print('Resources:')
        print('')
        for name in sorted(self._resources.keys()):
            methods = sorted(self._resources[name]._methods.keys())
            print('{}: {}'.format(bold(name), ', '.join(methods)))