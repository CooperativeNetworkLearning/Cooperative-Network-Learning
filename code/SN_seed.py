from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import pickle
##存放节点信息
import redis
redis_config = {
    'service_ip' : '127.0.0.1',
    'port' : 6379
}
class S(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        paths = {
            '/foo': {'status': 200},
            '/bar': {'status': 302},
            '/baz': {'status': 404},
            '/qux': {'status': 500}
        }

        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({'status': 500})
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself

        ##post_data 为 传入SN节点信息，包括name,ip,port
        SN_info = pickle.loads(post_data)
        if isinstance(SN_info, dict):
            logging.info("get request from %s"%SN_info['name'])
            #将序列化文件存入redis
            self.r.sadd('sn_list',post_data)
            #获取已有sn序列的集合
            node_list = self.r.smembers('sn_list')
            logging.info(SN_info)
            #将集合再序列化作为结果传出
            res = pickle.dumps(node_list)
            self.do_HEAD()
            # self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
            # print("{}".format(res).encode('utf-8'))
            # self.wfile.write("{}".format(res).encode('utf-8'))
            self.wfile.write(res)

    def respond(self, opts):
        response = self.handle_http(opts['status'], self.path)
        self.wfile.write(response)

    def handle_http(self, status_code, path):
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = '''
           <html><head><title>Title goes here.</title></head>
           <body><p>This is a test.</p>
           <p>You accessed path: {}</p>
           </body></html>
           '''.format(path)
        return bytes(content, 'UTF-8')
    def set_redis(self, redis_config):
        self.r = redis.StrictRedis(host=redis_config['service_ip'], port=redis_config['port'], db=0)
        self.r.delete('sn_list')


def run(server_class=HTTPServer, handler_class=S, port=8080):
    print("run()")
    handler_class.set_redis(handler_class,redis_config = redis_config)
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("httpd.server_close()")
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()