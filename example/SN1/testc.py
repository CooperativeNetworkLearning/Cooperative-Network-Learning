import zerorpc

class HelloRPC(object):
    def hello(self, name):
        return "Hello, %s" % name
    def hello1(self):
        return
        pass
s = zerorpc.Server(HelloRPC())
s.bind("tcp://0.0.0.0:4242")
s.run()
