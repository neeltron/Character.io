from flask import Flask, render_template
app = Flask('app', template_folder = "templates", static_folder = "gen_data")

@app.route('/')
def characterio():
  return 'Hello, World!'

app.run(host='0.0.0.0', port=8080)
