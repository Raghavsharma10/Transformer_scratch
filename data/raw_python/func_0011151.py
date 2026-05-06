def create_birthday(min_age=18, max_age=80):
    """
    Create a random birthday fomr someone between the ages of min_age and max_age
    """
    age = random.randint(min_age, max_age)
    start = datetime.date.today() - datetime.timedelta(days=random.randint(0, 365))
    return start - datetime.timedelta(days=age * 365)