class FormattedList:
    def __init__(self, data):
        self.data = data
    def __repr__(self):
        return "\n".join(self.data)