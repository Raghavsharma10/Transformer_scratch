def requirements():
  """Build the requirements list for this project"""
  requirements_list = []
  with open('requirements.txt') as requirements:
    for install in requirements:
      requirements_list.append(install.strip())
  return requirements_list