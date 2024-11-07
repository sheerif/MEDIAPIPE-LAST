#!/usr/bin/python

import tkinter as tk
import cv2
import logging
import numpy as np
import time
import socket
import subprocess
from PIL import Image, ImageTk
from datetime import datetime
from openni import openni2
from openni import _openni2 as c_api
from logging.config import dictConfig

# Paramètres et variables globales
dev = None
color_stream = None
Repertoire_Sauvegarde = "/home/Share/Enregistrements/"
video_duration = 120  # Durée d'enregistrement en secondes (2 minutes)

# Configuration du logging
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {'format': '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'}
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': logging.DEBUG
        }
    },
    'root': {
        'handlers': ['console'],
        'level': logging.DEBUG,
    },
}
dictConfig(logging_config)
logger = logging.getLogger('Device')
logger.debug("Rerouting stdout,stderr to logger")

# Fonctions
def connect_cam():
    global dev
    try:
        openni2.initialize()
        dev = openni2.Device.open_any()
        logger.debug("Caméra connectée.")
    except Exception as e:
        logger.error(f"Erreur de connexion à la caméra: {e}")

def disconnect_cam():
    global color_stream
    if color_stream:
        color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()

def test_cam():
    txt_checkCam.set("NO CAM")
    try:
        result = subprocess.check_output("lsusb | grep Orbbec", shell=True).decode()
        txt_checkCam.set(result if result else "NO CAM")
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la caméra: {e}")
        txt_checkCam.set("Erreur de vérification")

def recording():
    global color_stream
    try:
        # Démarrer le flux de couleur
        color_stream = dev.create_color_stream()
        color_stream.start()
        color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))

        # Configurer l'enregistrement vidéo
        now = datetime.now()
        date_str = now.strftime("%d_%m_%Y_%H_%M_%S")
        video_filename = f"{Repertoire_Sauvegarde}video_{date_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Utilisation de mp4v pour le format MP4
        out = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))

        logger.info(f"Démarrage de l'enregistrement vidéo : {video_filename}")

        start_time = time.time()
        while True:
            # Lire un nouveau cadre de couleur
            color_frame = color_stream.read_frame()
            color_frame_data = color_frame.get_buffer_as_uint8()
            color_img = np.frombuffer(color_frame_data, dtype=np.uint8).reshape((480, 640, 3))[..., ::-1]

            # Écrire le cadre dans le fichier vidéo
            out.write(color_img)

            # Afficher l'image pour le retour visuel (facultatif)
            cv2.imshow("Video Stream", color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Vérifier la durée d'enregistrement
            if time.time() - start_time > video_duration:
                break

        out.release()  # Libérer l'objet VideoWriter
        logger.info(f"Vidéo enregistrée avec succès : {video_filename}")

    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement de la vidéo: {e}")
    finally:
        disconnect_cam()

def prise_photo():
    connect_cam()
    recording()

def update_last_image(last_filename=None):
    if last_filename is None:
        with open("Last_img.txt", "r") as fichier:
            last_filename = fichier.read().strip()
    txt_last_img.set(last_filename)
    img_cv2 = cv2.imread(last_filename)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    Label_image.config(image=img_tk)
    Label_image.image = img_tk

# Interface graphique
MAIN = tk.Tk()
txt_last_img = tk.StringVar()
txt_checkCam = tk.StringVar()
Label_image = tk.Label(MAIN)

MAIN.title("Cabine Connectée - Utilitaire de Diagnostic")
nomPoste = f"POSTE : {socket.gethostname()}"
Label_nomPoste = tk.Label(MAIN, text=nomPoste, fg="red", bg=None, font=("Courrier", 10))
Label_nomPoste.pack()

BP_checkCam = tk.Button(MAIN, text="VERIFIER LA CAMERA", command=test_cam, fg="black", bg="white")
BP_checkCam.pack()

label_checkCam = tk.Label(MAIN, textvariable=txt_checkCam, fg="black", bg="white")
label_checkCam.pack()

BP_prise_photo = tk.Button(MAIN, text="Prendre une vidéo de 2 minutes", command=prise_photo, fg="black", bg="white")
BP_prise_photo.pack()

update_last_image()
Label_image.pack()

label_last_img = tk.Label(MAIN, textvariable=txt_last_img, fg="black", bg="white")
label_last_img.pack()

BP_fermer = tk.Button(MAIN, text="QUITTER", command=MAIN.quit, fg="red", bg="white")
BP_fermer.pack()

try:
    MAIN.mainloop()
except KeyboardInterrupt:
    logger.info("Interruption par l'utilisateur, fermeture du programme.")
    disconnect_cam()  # Assurez-vous de libérer les ressources
finally:
    logger.info("Programme terminé.")
