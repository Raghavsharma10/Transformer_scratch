def Parse(self, value):
    """Parse a 'Value' declaration.

    Args:
      value: String line from a template file, must begin with 'Value '.

    Raises:
      TextFSMTemplateError: Value declaration contains an error.

    """

    value_line = value.split(' ')
    if len(value_line) < 3:
      raise TextFSMTemplateError('Expect at least 3 tokens on line.')

    if not value_line[2].startswith('('):
      # Options are present
      options = value_line[1]
      for option in options.split(','):
        self._AddOption(option)
      # Call option OnCreateOptions callbacks
      [option.OnCreateOptions() for option in self.options]

      self.name = value_line[2]
      self.regex = ' '.join(value_line[3:])
    else:
      # There were no valid options, so there are no options.
      # Treat this argument as the name.
      self.name = value_line[1]
      self.regex = ' '.join(value_line[2:])

    if len(self.name) > self.max_name_len:
      raise TextFSMTemplateError(
          "Invalid Value name '%s' or name too long." % self.name)

    if (not re.match(r'^\(.*\)$', self.regex) or
        self.regex.count('(') != self.regex.count(')')):
      raise TextFSMTemplateError(
          "Value '%s' must be contained within a '()' pair." % self.regex)

    self.template = re.sub(r'^\(', '(?P<%s>' % self.name, self.regex)