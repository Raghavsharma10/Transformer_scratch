def parse_module_class(self):
        """Parse the module and class name part of the fully qualifed class name.

        """
        cname = self.class_name
        match = re.match(self.CLASS_REGEX, cname)
        if not match:
            raise ValueError(f'not a fully qualified class name: {cname}')
        return match.groups()