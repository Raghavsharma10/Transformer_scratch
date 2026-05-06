def add(name, function_name, cron):
    """ Create an event """
    lambder.add_event(name=name, function_name=function_name, cron=cron)