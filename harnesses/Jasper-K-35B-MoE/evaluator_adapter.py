#!/usr/bin/env python3
"""HTTP adapter for a reader-provided evaluator; contains no evaluation backend."""
from __future__ import annotations

import argparse
import importlib
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def validate_request(request: dict) -> None:
    if not isinstance(request, dict):
        raise ValueError('request must be a JSON object')
    for key in ('ref_code', 'kernel_code'):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    if not isinstance(request.get('cfg'), dict):
        raise ValueError('cfg must be an object')
    timeout = request.get('timeout')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')


def validate_result(result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError('evaluator must return a dictionary')
    for key in ('compiled', 'correctness'):
        if type(result.get(key)) is not bool:
            raise ValueError(f'{key} must be a boolean')
    for key in ('cheated', 'infra_error'):
        if key in result and type(result[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    if result['correctness']:
        if not result['compiled']:
            raise ValueError('a correct result must also be compiled')
        speedup = result.get('speedup')
        if type(speedup) not in (int, float) or not math.isfinite(speedup) or speedup <= 0:
            raise ValueError('a correct result requires a measured, positive speedup')
    json.dumps(result, allow_nan=False)
    return result


def make_server(evaluate, host='127.0.0.1', port=9000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload, status=200):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip('/') == '/health':
                self.respond({'status': 'ok', 'service': 'Jasper-K evaluator adapter',
                              'backend_validation': 'not performed by health check'})
            else:
                self.respond({'error': 'not found'}, 404)

        def do_POST(self):
            if self.path.rstrip('/') != '/evaluate':
                self.respond({'error': 'not found'}, 404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError('request size must be between 1 byte and 16 MiB')
                request = json.loads(self.rfile.read(length))
                validate_request(request)
            except (ValueError, UnicodeDecodeError) as exc:
                self.respond({'error': str(exc)}, 400)
                return
            try:
                # The plugin owns execution isolation, GPU scheduling, and timeout.
                result = validate_result(evaluate(request))
            except NotImplementedError as exc:
                self.respond({'error': str(exc), 'infra_error': True}, 501)
                return
            except Exception as exc:
                self.respond({'error': f'{type(exc).__name__}: {exc}', 'infra_error': True}, 500)
                return
            self.respond(result)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluator', required=True, help='Python module:function, e.g. my_evaluator:evaluate')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    module, separator, function = args.evaluator.partition(':')
    if not separator:
        parser.error('--evaluator must be module:function')
    evaluate = getattr(importlib.import_module(module), function)
    if not callable(evaluate):
        parser.error('the evaluator entry point must be callable')
    server = make_server(evaluate, args.host, args.port)
    print(f'Jasper-K evaluator adapter: http://{args.host}:{server.server_address[1]}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
