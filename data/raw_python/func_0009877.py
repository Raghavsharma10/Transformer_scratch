def Security_setOverrideCertificateErrors(self, override):
		"""
		Function path: Security.setOverrideCertificateErrors
			Domain: Security
			Method name: setOverrideCertificateErrors
		
			Parameters:
				Required arguments:
					'override' (type: boolean) -> If true, certificate errors will be overridden.
			No return value.
		
			Description: Enable/disable overriding certificate errors. If enabled, all certificate error events need to be handled by the DevTools client and should be answered with handleCertificateError commands.
		"""
		assert isinstance(override, (bool,)
		    ), "Argument 'override' must be of type '['bool']'. Received type: '%s'" % type(
		    override)
		subdom_funcs = self.synchronous_command(
		    'Security.setOverrideCertificateErrors', override=override)
		return subdom_funcs