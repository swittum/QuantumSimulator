class NormZeroError(Exception):
    def __init__(self):
        message = 'State has norm zero, normalization not possible.'
        super().__init__(message)
