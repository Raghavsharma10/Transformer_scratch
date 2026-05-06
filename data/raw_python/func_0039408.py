def task_coverage():
    """show coverage for all modules including tests"""
    cov = Coverage(
        [PythonPackage('import_deps', 'tests')],
        config={'branch':True,},
    )
    yield cov.all() # create task `coverage`
    yield cov.src()