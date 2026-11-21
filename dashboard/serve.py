"""Веб-дашборд результатов. Только стандартная библиотека (zero-dependency).

Запуск:  python3 dashboard/serve.py  ->  http://localhost:8000
"""
import os
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main(port=8000):
    addr = ("127.0.0.1", port)
    with TCPServer(addr, Handler) as httpd:
        print(f"Dashboard: http://{addr[0]}:{addr[1]}")
        httpd.serve_forever()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
