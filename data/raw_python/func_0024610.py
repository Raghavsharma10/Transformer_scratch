def BitmathType(bmstring):
    """An 'argument type' for integrations with the argparse module.

For more information, see
https://docs.python.org/2/library/argparse.html#type Of particular
interest to us is this bit:

   ``type=`` can take any callable that takes a single string
   argument and returns the converted value

I.e., ``type`` can be a function (such as this function) or a class
which implements the ``__call__`` method.

Example usage of the bitmath.BitmathType argparser type:

   >>> import bitmath
   >>> import argparse
   >>> parser = argparse.ArgumentParser()
   >>> parser.add_argument("--file-size", type=bitmath.BitmathType)
   >>> parser.parse_args("--file-size 1337MiB".split())
   Namespace(file_size=MiB(1337.0))

Invalid usage includes any input that the bitmath.parse_string
function already rejects. Additionally, **UNQUOTED** arguments with
spaces in them are rejected (shlex.split used in the following
examples to conserve single quotes in the parse_args call):

   >>> parser = argparse.ArgumentParser()
   >>> parser.add_argument("--file-size", type=bitmath.BitmathType)
   >>> import shlex

   >>> # The following is ACCEPTABLE USAGE:
   ...
   >>> parser.parse_args(shlex.split("--file-size '1337 MiB'"))
   Namespace(file_size=MiB(1337.0))

   >>> # The following is INCORRECT USAGE because the string "1337 MiB" is not quoted!
   ...
   >>> parser.parse_args(shlex.split("--file-size 1337 MiB"))
   error: argument --file-size: 1337 can not be parsed into a valid bitmath object
"""
    try:
        argvalue = bitmath.parse_string(bmstring)
    except ValueError:
        raise argparse.ArgumentTypeError("'%s' can not be parsed into a valid bitmath object" %
                                         bmstring)
    else:
        return argvalue