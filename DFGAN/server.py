import base64
import subprocess

from flask import Flask, request, jsonify
import cv2
import time



app = Flask(__name__)

def run_command(cmd):
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stdout, process.stderr

@app.route("/predict", methods=["POST"])
def predict():
    # Activate the conda environment
    start_time = time.time()
    # cmd = "activate torch_env"
    # stdout, stderr = run_command(cmd)
    # print("Environment activated")
    # print(stdout.decode(), stderr.decode())
    # cmd = "conda env list"
    # stdout, stderr = run_command(cmd)
    # #print("Environment activated")
    # print(stdout.decode(), stderr.decode())

    # Run the main.py script
    cmd = "python main.py"
    stdout, stderr = run_command(cmd)
    print("Main script run")
    # print(stdout.decode(), stderr.decode())

    # cmd = "deactivate"
    # stdout, stderr = run_command(cmd)
    # print("Environment Deactivated")
    # print(stdout.decode(), stderr.decode())
    # cmd = "conda env list"
    # stdout, stderr = run_command(cmd)
    # #print("Environment activated")
    # print(stdout.decode(), stderr.decode())

    # Load the generated image
    path = r'D:\BE_Major_Project\Projects\DFGAN\Output\generated_image.png'
    image = cv2.imread(path)
    
    # Return the generated image in the response
    retval, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    end_time = time.time()

    execution_time = end_time - start_time
    response = {
        "image": image_base64,"time": execution_time
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True,port=8000)