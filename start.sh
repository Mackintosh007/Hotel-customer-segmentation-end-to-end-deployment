#!/bin/sh
set -e

# Run the Flask application using the Python interpreter directly.
# We set the FLASK_APP environment variable to point to our app.py file
# and then use the `flask run` command.
# The `exec` command ensures that the process runs as the main container process.
exec python -m flask run --host=0.0.0.0
