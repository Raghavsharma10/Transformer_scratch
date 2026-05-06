def stub_main():
    """setuptools blah: it still can't run a module as a script entry_point"""
    from google.apputils import run_script_module
    import butcher.main
    run_script_module.RunScriptModule(butcher.main)