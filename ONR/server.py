import json
import os
from http import server
import numpy as np

import tensorflow as tf

import ONR.model as model


class JSONHandler(server.BaseHTTPRequestHandler):

    def do_POST(self):
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len)
        payload = json.loads(content)

        with tf.Session() as sess:
            nn = model.Convnet()
            # try:
            image = np.array(payload.get("image"))
            print(image)
            result = int(nn.predict(sess, image))
            print("Is the number %d ?" % result)
            response = {"result": result}
            # except:
            #     response_code = 500

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(bytes(json.dumps(response), 'utf-8'))

    def do_GET(self):  # works fine for browser to get the page source
        response_code = 200
        try:
            with open('index.html', 'rb') as f:
                self.send_response(response_code)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(f.read())
        except IOError:
            self.send_error(404, "Page Not Found")


if __name__ == "__main__":
    host = 'localhost'
    port = 8000
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    server_class = server.HTTPServer
    httpd = server_class((host, port), JSONHandler)
    print("Server ready")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server terminated")
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()
