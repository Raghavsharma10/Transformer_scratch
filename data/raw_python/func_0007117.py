def get_arguments(options):
   """ This function handles and validates the wrapper arguments. """
   # These the next couple of lines defines the header of the Help output
   parser = ArgumentParser(
      formatter_class=RawDescriptionHelpFormatter,
      usage=("""%(prog)s
--------------------------------------------------------------------------------
"""),
      description=("""
Service Wrapper
===============
This is the service wrapper script, which is a part of the CGE services.
Read the online manual for help.
A list of all published services can be found at:
cge.cbs.dtu.dk/services

"""), epilog=("""
--------------------------------------------------------------------------------
      """))

   #ADDING ARGUMENTS
   setarg = parser.add_argument
   #SERVICE SPECIFIC ARGUMENTS
   if isinstance(options, str):
      options = [[x for i,x in enumerate(line.split()) if i in [1,2]] for line in options.split('\n') if len(line)>0]
      for o in options:
         try:
            setarg(o[1], type=str, dest=o[0], default=None, help=SUPPRESS)
         except:
            None
   else:
      for o in options:
         if o[2] is True:
            # Handle negative flags
            setarg(o[0], action="store_false", dest=o[1], default=o[2],
                   help=o[3])
         elif o[2] is False:
            # Handle positive flags
            setarg(o[0], action="store_true", dest=o[1], default=o[2],
                   help=o[3])
         else:
            help_ = o[3] if o[2] is None else "%s [%s]"%(o[3], '%(default)s')
            setarg(o[0], type=str, dest=o[1], default=o[2],
                   help=help_)
   # VALIDATION OF ARGUMENTS
   args = parser.parse_args()
   debug.log("ARGS: %s"%args)
   return args