def overload(fn, function_to_overload=None):
        '''
This function decorator allows you to overload already defined functions. The
execution of overloaded functions is done by trying the original version first
and if it fails, the variables are handed off to the overloading function.

While this does seem like a sloppy way to go about choosing the execution of
functions, this gives you far more control in terms of how you want each
function to be selected and allows you to program for the "ideal situation"
first.

With this approach, you can simply require very specific conditions that would
apply to a majority of the use cases of the function and allow the code to
mitigate edge case scenarios only when the edge cases show up vs checking for
edge cases on every single usage of the function.

This approach rewards functions that are designed with proper input validation,
which you should be adding anyways.

#------------------------------------------------------------------------------
#   Example Usage Below
#------------------------------------------------------------------------------

def my_print(arg):
    print('running original my_print')
    print(arg)

@overload
def my_print(arg):
    assert type(arg) == list
    print('running list my_print')
    print(', '.join(str(i) for i in arg))

@overload
def my_print(arg):
    assert type(arg) == dict
    print('running dict my_print')
    out = ('='.join((str(k), str(v))) for k,v in arg.items())
    print(' | '.join(out))

my_print(list(range(10)))
# running list my_print
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

my_print(tuple(range(10)))
# running original my_print
# (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

my_print({i:i*2 for i in range(10)})
# running dict my_print
# 0=0 | 1=2 | 2=4 | 3=6 | 4=8 | 5=10 | 6=12 | 7=14 | 8=16 | 9=18

'''
        if function_to_overload is None:
            if Overload.has_args():
                return Overload.configured_decorator(fn)
            else:
                return Overload.default_decorator(fn)
        else:
            return Overload.configured_decorator(function_to_overload)(fn)