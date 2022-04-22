
import gc
import requests, sys, os

import torch
from generator_disco.generator import GeneratorDisco
from generator_ld.generator import GeneratorLatentDiffusion

from flask import Flask, flash, request, redirect, url_for, render_template,make_response,send_file,Response
from werkzeug.utils import secure_filename
from twilio.twiml.messaging_response import MessagingResponse, Message, Redirect, Body
from twilio.rest import Client 

from random import randint, seed

from manager.chain.chain import Chain

UPLOAD_FOLDER = 'static/uploads/'

os.system("export TOKENIZERS_PARALLELISM=false")

app = Flask(__name__)

app.secret_key = "1923493493"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['mp4', 'mp3', 'wav','.mov'])

chain = None

@app.route('/make/<prompt>', methods=['GET', 'POST'])
def make(prompt):
    filename = chain.run_chain(prompt)
    return send_file("static/output/" + filename, mimetype='image/png')

@app.route('/bot', methods=['POST'])
def bot():
    incoming_msg = request.values.get('Body', '').lower()
    print(incoming_msg)
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
   
    if 'make' in incoming_msg:
        input_seed = "" #str(100)
        prefix = str(randint(0,1000000))#--steps 50
        prompt = incoming_msg.replace("make ","")
        path_to_image = chain.run_chain(prompt) 
        msg.body(prompt)
        msg.media("https://ce1c-86-170-32-104.eu.ngrok.io/static/output/" + path_to_image)
        responded = True
        print ("constructed message")
    if not responded:
        msg.body('Meh!!')

    response_string = str(resp)
    print ("response", response_string)
    # response = MessagingResponse()
    # message = Message()
    # message.body('Hello World!')
    # response.append(message)
    # # return make_response(str(response))

    response = make_response(response_string)
    response.headers["Content-Type"] = "text/xml"
    return response


def load_chain():
    global chain
    chain = Chain()
    
if __name__ == '__main__':
    print("running")
    load_chain()
    app.run(debug=False,host = "0.0.0.0")
