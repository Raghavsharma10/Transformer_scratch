def get(self, request, *args, **kwargs):
        '''The queryset returns all measurement objects'''
        measurements = Measurement.objects.all()    
        return data_csv(self.request, measurements)