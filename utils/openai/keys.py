class Keys:
    def __init__(self, apikey=None):
        if apikey:
            self.apikeys = [apikey]
        else:
            with open('./utils/openai/apikeys.txt', 'r') as file:
                lines = file.readlines()

                # Exclude the first line (instruction message)
                self.apikeys = lines[1:]

        self.current_apikey = 0
        print(f"total {len(self.apikeys)} keys")

    def get_apikey(self):
        apikey = self.apikeys[self.current_apikey]
        self.current_apikey = (self.current_apikey + 1) % len(self.apikeys)
        return apikey
