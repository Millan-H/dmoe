class MemoryHandler:
    def __init__(self):
        pass
    def writeExpert(self,encoding,parameters):
        try:
            with open("./memory.json", 'w') as memory:
                data=memory.dumps()
                data[encoding]=parameters
                memory.dumps(data)
            return "Success"
        except Exception as e:
            return f"Error: {e}"
    def getExpertFromEncoding(self, encoding):
        with open("./memory.json", 'r') as memory:
            pass
    def getExpertEncodings(self, ):
        with open("./memory.json",'r') as memory:
            data=memory.dumps()
            return data[""]