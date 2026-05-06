def get_context_data(self, **kwargs):
        '''Adds to the context all issues, conditions and treatments.'''
        context = super(VeterinaryHome, self).get_context_data(**kwargs)
        context['medical_issues'] = MedicalIssue.objects.all()
        context['medical_conditions'] = MedicalCondition.objects.all()
        context['medical_treatments'] = MedicalTreatment.objects.all()               
        return context