def has_args():
        ''' returns true if the decorator invocation
            had arguments passed to it before being
            sent a function to decorate '''
        no_args_syntax = '@overload'
        args_syntax = no_args_syntax + '('
        args, no_args = [(-1,-1)], [(-1,-1)]
        for i, line in enumerate(Overload.traceback_lines()):
            if args_syntax in line:
                args.append((i, line.find(args_syntax)))
            if no_args_syntax in line:
                no_args.append((i, line.find(no_args_syntax)))
        args, no_args = max(args), max(no_args)
        if sum(args)+sum(no_args) == -4:
            # couldnt find invocation
            return False
        return args >= no_args