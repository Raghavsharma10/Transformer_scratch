def check(module):
  global passed, failed
  '''
  apply pylint to the file specified if it is a *.py file
  '''
  module_name = module.rsplit('/', 1)[1]
  if module[-3:] == ".py" and module_name not in IGNORED_FILES:
    print ("CHECKING ", module)
    pout = os.popen('pylint %s'% module, 'r')
    for line in pout:
      if "Your code has been rated at" in line:
        print ("PASSED pylint inspection: " + line)
        passed += 1
        return True
      if "-error" in line:
        print ("FAILED pylint inspection: " + line)
        failed += 1
        errors.append("FILE: " + module)
        errors.append("FAILED pylint inspection: " + line)
        return False