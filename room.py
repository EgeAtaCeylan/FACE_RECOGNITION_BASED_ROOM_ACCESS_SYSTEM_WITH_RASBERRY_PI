import os
import socket
import picamera
import time
import io
import struct
from datetime import datetime

# IP address of the server machine that performs face recognition.
# This must be replaced with the real server IP before running the client.
server_Ip = "Your server Ip"

# Create the client socket and connect to the face recognition server.
cSocket = socket.socket()
cSocket.connect((server_Ip, 9000))

# Create a file-like socket wrapper for writing binary image data.
connection = cSocket.makefile('wrb')

# Start the Raspberry Pi camera.
camera = picamera.PiCamera()
print("Welcome to the room, please DIRECTLY look at the camera to enter inside")

# Camera settings.
camera.vflip = False
camera.resolution = (500, 480)

# Show a small preview window so the user can position themselves.
camera.start_preview(fullscreen=False, window=(100, 200, 800, 900))

# Countdown before the recognition process begins.
print("Entering process will start in")
for i in range(5, 0, -1):
    print(i)
    time.sleep(1)

# Record the start time so the system can stop after a maximum retry period.
startingTime = time.time()

try:
    # In-memory stream used to hold each captured JPEG frame.
    stream = io.BytesIO()

    # Continuously capture frames from the Pi camera.
    for foo in camera.capture_continuous(stream, 'jpeg'):
        # First send the frame length so the server knows how many bytes to read.
        connection.write(struct.pack('<L', stream.tell()))
        connection.flush()

        # Move back to the start of the stream and send the actual JPEG bytes.
        stream.seek(0)
        connection.write(stream.read())

        # Reset the stream so it can be reused for the next captured frame.
        stream.seek(0)
        stream.truncate()

        # Wait for the server's decision message.
        recievedMessage = cSocket.recv(1024).decode()
        splittedMessage = recievedMessage.split()
        code = splittedMessage[0]
        person = splittedMessage[1]

        # code == 0 means access is denied, but there are multiple denial reasons.
        if code == str(0):
            # person == -1 means no face was detected.
            if person == str(-1):
                print("No face is detected in the camera")
                print("Please DIRECTLY look at the camera again")
                print("Re-trying in")
                for i in range(5, 0, -1):
                    print(i)
                    time.sleep(1)

            # person == 0 means the face was unknown to the model.
            elif person == str(0):
                print("You don't have access to this room")
                print("Please DIRECTLY look at the camera again")
                print("Re-trying in")
                for i in range(5, 0, -1):
                    print(i)
                    time.sleep(1)

            # Otherwise the server returned a recognized name with denied access.
            else:
                print(person, " you don't have access to the room")
                print(person, " please DIRECTLY look at the camera again")
                print("Re-trying in")
                for i in range(5, 0, -1):
                    print(i)
                    time.sleep(1)

        # code == 1 means the user is recognized and authorized.
        elif code == str(1):
            print(person, " welcome to the room. Have a nice day")
            break

        # Any other code in this project means multiple faces were detected.
        else:
            print("More than 1 face detected in the camera")
            print("PLEASE SHOW ONLY 1 (ONE) FACE TO THE CAMERA")
            print("Re-trying in")
            for i in range(5, 0, -1):
                print(i)
                time.sleep(1)

        # Stop retrying if the process has taken longer than one minute.
        if time.time() - startingTime > 60:
            print("Maximum trying time exceeded, TERMINATING THE ENTERING PROCESS")
            break

    # Send a final zero-length frame to tell the server the client is done.
    connection.write(struct.pack('<L', 0))

finally:
    # Always close the socket before exiting.
    cSocket.close()
