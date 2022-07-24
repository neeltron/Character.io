from flask import Flask, render_template
import random

app = Flask('app', template_folder = "templates", static_folder = "gen_data")

@app.route('/')
def characterio():
  rando = random.randint(1, 250)
  return render_template('index.html', data = rando)

app.run(host='0.0.0.0', port=8080)
