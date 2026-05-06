def get_auth_error_message(self):
        ''' create an informative error message if there is an issue authenticating'''
        errors = ["Authentication error retrieving ec2 inventory."]
        if None in [os.environ.get('AWS_ACCESS_KEY_ID'), os.environ.get('AWS_SECRET_ACCESS_KEY')]:
            errors.append(' - No AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY environment vars found')
        else:
            errors.append(' - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment vars found but may not be correct')

        boto_paths = ['/etc/boto.cfg', '~/.boto', '~/.aws/credentials']
        boto_config_found = list(p for p in boto_paths if os.path.isfile(os.path.expanduser(p)))
        if len(boto_config_found) > 0:
            errors.append(" - Boto configs found at '%s', but the credentials contained may not be correct" % ', '.join(boto_config_found))
        else:
            errors.append(" - No Boto config found at any expected location '%s'" % ', '.join(boto_paths))

        return '\n'.join(errors)