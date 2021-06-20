from core.main import get_response_bot
from flask import Flask
app = Flask(__name__)

@app.route('/')
def welcome():
	return 'Bonjour, je suis là pour vous aidez à vous rediriger vers un spécialiste, où avez-vous mal ?'

@app.route('/message/<message>')
def question(message):
	return get_response_bot(message)

if __name__ == '__main__':
	app.debug = False
	app.run(port=4000)
