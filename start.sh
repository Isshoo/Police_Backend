#!/bin/bash
exec gunicorn --worker-class gevent \
              --workers 2 \
              --bind 0.0.0.0:$PORT \
              --timeout 120 \
              --access-logfile - \
              --error-logfile errors.log \
              "app:create_app()"