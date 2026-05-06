def get_template(name, base=None):
    '''read in and return a template file
    '''
    # If the file doesn't exist, assume relative to base
    template_file = name
    if not os.path.exists(template_file):
        if base is None:
            base = get_templatedir()
        template_file = "%s/%s" %(base, name)

    # Then try again, if it still doesn't exist, bad name
    if os.path.exists(template_file):
        with open(template_file,"r") as filey:
            template = "".join(filey.readlines())
        return template
    bot.error("%s does not exist." %template_file)