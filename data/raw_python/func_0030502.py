def run(self):
    """Execute the build command."""

    module = self.distribution.ext_modules[0]

    building_for_windows = self.plat_name in ['win32', 'win-amd64']
    if building_for_windows:
      module.define_macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
      module.libraries.append('advapi32')

    if self.dynamic_linking:
      module.libraries.append('yara')
    else:
      for source in yara_sources:
        module.sources.append(source)

    build_ext.build_ext.run(self)