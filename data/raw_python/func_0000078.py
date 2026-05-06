def license_loader(lic_dir=LIC_DIR):
    """Loads licenses from the given directory."""
    lics = []
    for ln in os.listdir(lic_dir):
        lp = os.path.join(lic_dir, ln)
        with open(lp) as lf:
            txt = lf.read()
            lic = License(txt)
            lics.append(lic)
    return lics