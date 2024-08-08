class DataHandler:
    def __init__(self) -> None:
        self.data = {}

    def add_data(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            # Check if the value is a list
            if isinstance(value, list):
                self.data[key].extend(value)  # Extend the existing list
            else:
                self.data[key].append(value)  # Append non-list values

    def clear_data(self):
        self.data = {}