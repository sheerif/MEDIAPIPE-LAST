#!/usr/bin/python
# -*- coding: utf-8 -*-

# =============================================
# Importations
# =============================================
import os
import sys
import subprocess
import threading
import time
import socket
import logging
import signal
from datetime import datetime
from math import degrees, acos
from typing import List
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp
import paramiko
from termcolor import colored
from openni import openni2
from openni import _openni2 as c_api
import platform
import faulthandler
import gc
from collections import namedtuple
import argparse

# =============================================
# Informations de Version
# =============================================
__version__ = "1.0.0"

# =============================================
# Gestion de la journalisation sans remplacer stdout/stderr
# =============================================

# Activer faulthandler pour obtenir un traceback en cas de segfault
faulthandler.enable()

# Désactiver le garbage collector pour éviter les interférences potentielles
gc.disable()

# Configuration de la journalisation avec FileHandler et StreamHandler
logging.basicConfig(
    level=logging.DEBUG,  # Niveau de log défini à DEBUG pour le maximum d'informations
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("application.log"),  # Enregistrement des logs dans un fichier
        logging.StreamHandler(sys.stdout)       # Affichage des logs sur la console
    ]
)

logging.info("Configuration du logging réussie.")
logging.info(f"Début de l'application CameraApplication, Version {__version__}")

# =============================================
# Définition des namedtuples
# =============================================

# Exemple de namedtuple pour stocker les points clés
KeyPoint = namedtuple('KeyPoint', ['x', 'y'])

# =============================================
# Définition des KeyPoints avec Enum
# =============================================

class KeyPoints(Enum):
    CHEST_FRONT = 0
    CHEST_ROTATION = 1
    CHEST_SIDE = 2
    ELBOW_LEFT = 3
    ELBOW_RIGHT = 4
    SHOULDER_LEFT_ROTATION = 5
    SHOULDER_LEFT_RAISING = 6
    SHOULDER_RIGHT_ROTATION = 7
    SHOULDER_RIGHT_RAISING = 8
    NECK_FLEXION = 9

# =============================================
# Manager de KeyPoints avec thread-safety
# =============================================

class KeyPointManager:
    """Classe pour gérer les KeyPoints de manière thread-safe."""

    def __init__(self, lock: threading.Lock):
        self.lock = lock
        self.keypoints = []

    def add_keypoint(self, x: float, y: float):
        """Ajoute un KeyPoint de manière thread-safe."""
        with self.lock:
            kp = KeyPoint(x, y)
            self.keypoints.append(kp)
            logging.debug(f"KeyPoint ajouté: {kp}")

    def get_keypoints(self) -> List[KeyPoint]:
        """Retourne une copie des KeyPoints."""
        with self.lock:
            return list(self.keypoints)

# =============================================
# Classe Principale
# =============================================

class CameraApplication:
    """Classe principale pour gérer l'application de la caméra."""

    def __init__(self):
        # Ignorer le signal SIGPIPE pour éviter que le processus ne se termine brusquement
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        # ***** Variables/Paramètres *****
        self.plage_ip = "10.10.10."
        self.ip_concentrateur = 70
        self.full_ip_concentrateur = self.plage_ip + str(self.ip_concentrateur)
        
        self.color_img = None
        self.color = True
        self.tps_traitement = 0.4
        self.delai_pause = 4.0 - self.tps_traitement
        self.repertoire_sauvegarde = "/home/Share/Enregistrements/"
        self.mdv = 0
        self.nb_images_rec = 0
        self.nom_poste = socket.gethostname()
        self.num_poste = self.nom_poste.replace("pc-camera", "")
        self.nom_du_poste = f"Caméra n° : {self.num_poste}"
        self.table_srv = [""] * 6
        self.app_is_on = "no"
        self.recording = "no"
        self.pres_cam = "no"
        self.result_analyse = "_0_1_2_3_4_5_6_7_8_9"
        
        # Variables de synchronisation
        self.stop_event = threading.Event()
        self.periodic_thread = None

        # Paramètres de reconnexion
        self.max_retries = 5
        self.retry_delay = 5  # secondes

        # ***** Ajout des Attributs pour le Suivi des Scores *****
        self.action_history = []  # Historique des actions avec timestamps
        self.repetitivite_window = 60  # Fenêtre de temps en secondes pour le calcul

        self.posture_history = []  # Historique des postures avec timestamps
        self.posture_window = 60  # Fenêtre de temps en secondes pour le calcul

        self.last_activity_time = None
        self.recuperation_threshold = 30  # Temps de récupération en secondes

        self.hand_positions_history = []  # Historique des positions des poignets avec timestamps
        self.prehension_window = 60  # Fenêtre de temps en secondes pour le calcul

        # ***** Attributs pour la Gestion Dynamique de la Complexité du Modèle *****
        self.false_positives = 0
        self.false_negatives = 0
        self.performance_history = []  # Historique des performances avec timestamps
        self.performance_window = 60  # Fenêtre de temps en secondes pour le calcul des performances
        self.adjustment_threshold = 5  # Nombre d'erreurs pour ajuster la complexité

        # ***** Lock pour la Sécurité des Threads *****
        self.camera_lock = threading.Lock()
        self.mediapipe_lock = threading.Lock()  # Lock pour Mediapipe
        self.keypoint_lock = threading.Lock()   # Lock pour KeyPoint namedtuples

        # ***** Initialisation de Mediapipe Holistic *****
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2  # Initialement haut pour plus de précision
        )

        logging.info("*************** NE PAS FERMER ***************")
        logging.info("********** INITIALISATION **********")

        # ***** Initialisation du Manager de KeyPoints *****
        self.keypoint_manager = KeyPointManager(self.keypoint_lock)

    # ***** Fonction de Gestion des Signaux pour Arrêt Gracieux *****
    def signal_handler(self, sig, frame) -> None:
        """Gestionnaire de signaux pour un arrêt gracieux."""
        logging.info('Signal reçu. Fermeture en douceur...')
        self.sortie_programme()

    # ***** Fonction de Ping Améliorée *****
    def ping_host(self, host: str) -> bool:
        """Vérifie la connectivité avec un hôte en envoyant un ping."""
        try:
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            command = ['ping', param, '1', host]
            logging.info(f"Envoi d'un ping à {host}...")
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                logging.info(f"Ping réussi vers {host}.")
                return True
            else:
                logging.warning(f"Ping échoué vers {host}.")
                return False
        except Exception as e:
            logging.error(f"ping_host() - Exception: {e}")
            return False

    # ***** Initialisation de AnywhereUSB *****
    def init_anyusb(self) -> None:
        """Initialise et redémarre le périphérique AnywhereUSB."""
        try:
            ip_anyusb = 60 if (1 <= int(self.num_poste) <= 8) else 61
            portusb = int(self.num_poste) if (1 <= int(self.num_poste) <= 8) else int(self.num_poste) - 8
            
            hostname = f'10.10.10.{ip_anyusb}'
            username = os.getenv('SSH_USERNAME', 'admin')  # Utiliser des variables d'environnement
            password = os.getenv('SSH_PASSWORD', 'Masternaute2023*')  # Utiliser des variables d'environnement

            logging.info(f"Connexion SSH à {hostname} sur le port USB {portusb}...")
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname, username=username, password=password)
            
            commande = f'sudo system anywhereusb powercycle port{portusb}'
            logging.info(f"Exécution de la commande: {commande}")
            stdin, stdout, stderr = client.exec_command(commande)
            stdout.channel.recv_exit_status()  # Attendre la fin de la commande
            logging.info("Reboot caméra en cours...")
            client.close()
            time.sleep(10)
            logging.info("Redémarrage de la caméra FAIT!")
            
        except Exception as e:
            logging.error(f"init_anyusb() - Exception: {e}")
            try:
                client.close()
            except:
                pass

    # ***** Vérification de la Présence de la Caméra *****
    def check_cam(self) -> int:
        """Vérifie si la caméra est détectée sur le système."""
        try:
            command1 = 'lsusb | grep "Orbbec" > list_cam.txt'
            command2 = 'cp list_cam.txt /home/Share/list_cam.txt'
            logging.info("Exécution des commandes de vérification de la caméra...")
            self.cmd_terminal_local(command1)
            self.cmd_terminal_local(command2)

            list_cam = "/home/Share/list_cam.txt"
            with open(list_cam, "r") as f:
                content = f.read()
            cam_or_not = len(content)
            if cam_or_not == 0:
                logging.warning(colored("           Pas de caméra détectée ", "red"))
                self.init_anyusb()
            else:
                logging.info(colored("           Caméra détectée  ", "green"))

            return cam_or_not
        except Exception as e:
            logging.error(f"check_cam() - Exception: {e}")
            return 0

    # ***** Gestion du MDV *****
    def mdv_app(self) -> None:
        """Incrémente le compteur MDV."""
        try:
            self.mdv = (self.mdv + 1) % 60
            logging.debug(f"MDV mis à jour: {self.mdv}")
        except Exception as e:
            logging.error(f"mdv_app() - Exception: {e}")

    # ***** Enregistrement de l'Image *****
    def record_image(self) -> str:
        """Enregistre une image à partir du flux de la caméra."""
        try:
            with self.camera_lock:
                logging.info("Lecture d'une frame depuis le flux couleur...")
                color_frame = self.color_stream.read_frame()
                color_frame_data = color_frame.get_buffer_as_uint8()
                color_img = np.frombuffer(color_frame_data, dtype=np.uint8)
                
                expected_size = 480 * 640 * 3  # Ajustez selon la résolution de votre caméra
                logging.debug(f"Taille des données de frame: {len(color_frame_data)}")
                if len(color_frame_data) < expected_size:
                    logging.error("Données de frame insuffisantes pour le reshaping.")
                    return ""
                
                color_img = color_img.reshape((480, 640, 3))
                logging.debug(f"Forme de color_img après reshaping: {color_img.shape}")
                
                # Validation de la forme de l'image
                if color_img.shape != (480, 640, 3):
                    logging.error(f"Forme de l'image incorrecte après reshaping: {color_img.shape}")
                    return ""
                if not np.all((color_img >= 0) & (color_img <= 255)):
                    logging.error("Valeurs de l'image hors limites après conversion.")
                    return ""
                
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                
                now = datetime.now()
                date = now.strftime("%d_%m_%Y_%H_%M_%S_%f")
                filename = os.path.join(self.repertoire_sauvegarde, f"img_{date}.jpg")
                
                cv2.imwrite(filename, color_img)
                logging.debug(f"Image enregistrée: {filename}")
                with open("Last_img.txt", "w") as fichier:
                    fichier.write(filename)
                return filename
        except Exception as e:
            logging.error(f"record_image() - Exception: {e}")
            return ""

    # ***** Sortie du Programme *****
    def sortie_programme(self) -> None:
        """Ferme proprement toutes les ressources et quitte le programme."""
        try:
            # Signaler aux threads de s'arrêter
            self.stop_event.set()
            
            if self.periodic_thread and self.periodic_thread.is_alive():
                self.periodic_thread.join(timeout=5)
                logging.info("Thread périodique arrêté.")
            
            if hasattr(self, 'color_stream') and self.color_stream and self.color_stream.is_valid:
                logging.info("Arrêt du flux couleur...")
                self.color_stream.stop()
                logging.info("Flux couleur arrêté.")
            if hasattr(self, 'dev') and self.dev:
                logging.info("Déchargement de OpenNI...")
                openni2.unload()
                logging.info("OpenNI déchargé.")
            if hasattr(self, 'holistic') and self.holistic:
                logging.info("Fermeture de Mediapipe Holistic...")
                with self.mediapipe_lock:
                    self.holistic.close()
                logging.info("Mediapipe Holistic fermé.")
            cv2.destroyAllWindows()
            logging.info("Fenêtres OpenCV fermées.")
        except Exception as e:
            logging.error(f"sortie_programme() - Exception: {e}")
        finally:
            sys.exit(0)

    # ***** Exécution de Commandes Terminal Locales *****
    def cmd_terminal_local(self, command: str) -> None:
        """Exécute une commande dans le terminal local."""
        try:
            logging.info(f"Exécution de la commande: {command}")
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"cmd_terminal_local() - CalledProcessError: command = {command}, error = {e}")
        except Exception as e:
            logging.error(f"cmd_terminal_local() - Exception: command = {command}, error = {e}")

    # =============================================
    # Fonctions Utilitaires
    # =============================================

    def calculate_image_quality(self, image: np.ndarray) -> float:
        """
        Calculer la qualité de l'image en fonction de la luminosité et de la netteté.
        Utilisé pour ajuster la complexité du modèle MediaPipe.
        """
        if image is None or not isinstance(image, np.ndarray):
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 100.0
        quality = min(1.0, max(0.0, 0.5 * brightness + 0.5 * (sharpness / (sharpness + 1))))
        logging.debug(f"Qualité de l'image: {quality}")
        return quality

    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> int:
        """
        Calculer l'angle en degrés formé par trois points a, b, c.
        L'angle est au point b entre les segments ba et bc.
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        dot_product = np.dot(ba, bc)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return 0
        cos_angle = dot_product / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = int(degrees(acos(cos_angle)))
        logging.debug(f"Angle calculé: {angle} degrés")
        return angle

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraiter l'image pour améliorer la détection des points clés.
        """
        if image is None or not isinstance(image, np.ndarray):
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        processed_image = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        logging.debug("Image prétraitée.")
        return processed_image

    def classify_angle(self, angle: int, thresholds: dict) -> int:
        """
        Classer un angle donné dans une zone de risque selon les seuils fournis.
        """
        if thresholds['green'][0] <= angle <= thresholds['green'][1]:
            return 1
        elif any(lower <= angle <= upper for (lower, upper) in thresholds['orange']):
            return 2
        elif angle >= thresholds['red'][0]:
            return 3
        else:
            return 0

    def extract_keypoints(self, landmarks: List[mp.solutions.holistic.PoseLandmark]) -> dict:
        """
        Extraire les points clés nécessaires des landmarks détectés par Mediapipe.
        """
        def get_landmark_value(landmark):
            if landmark:
                return [landmark.x, landmark.y]
            else:
                return None

        keypoints = {
            KeyPoints.SHOULDER_LEFT_ROTATION: None,
            KeyPoints.ELBOW_LEFT: None,
            KeyPoints.WRIST_LEFT: None,
            KeyPoints.SHOULDER_RIGHT_ROTATION: None,
            KeyPoints.ELBOW_RIGHT: None,
            KeyPoints.WRIST_RIGHT: None,
            KeyPoints.NECK_FLEXION: None,
        }

        if landmarks:
            keypoints[KeyPoints.SHOULDER_LEFT_ROTATION] = get_landmark_value(landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER.value])
            keypoints[KeyPoints.ELBOW_LEFT] = get_landmark_value(landmarks[mp.solutions.holistic.PoseLandmark.LEFT_ELBOW.value])
            keypoints[KeyPoints.WRIST_LEFT] = get_landmark_value(landmarks[mp.solutions.holistic.PoseLandmark.LEFT_WRIST.value])
            keypoints[KeyPoints.SHOULDER_RIGHT_ROTATION] = get_landmark_value(landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER.value])
            keypoints[KeyPoints.ELBOW_RIGHT] = get_landmark_value(landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW.value])
            keypoints[KeyPoints.WRIST_RIGHT] = get_landmark_value(landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_WRIST.value])

            left_shoulder = keypoints[KeyPoints.SHOULDER_LEFT_ROTATION]
            right_shoulder = keypoints[KeyPoints.SHOULDER_RIGHT_ROTATION]

            if left_shoulder and right_shoulder:
                neck = [
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2
                ]
                keypoints[KeyPoints.NECK_FLEXION] = neck
            else:
                keypoints[KeyPoints.NECK_FLEXION] = None

        return keypoints

    def detect_actions_techniques_in_image(self, image: np.ndarray) -> List[tuple]:
        """
        Détecter des "actions techniques" dans l'image.
        """
        try:
            logging.info("Début de la détection des actions techniques dans l'image.")
            
            # Vérifier la forme et le type de l'image
            logging.debug(f"Forme initiale de l'image: {image.shape}")
            logging.debug(f"Type de données de l'image: {image.dtype}")
            
            if len(image.shape) == 3:
                logging.debug("Conversion de l'image en niveaux de gris.")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                logging.debug(f"Forme après conversion en niveaux de gris: {image.shape}")
            
            image_normalized = image / 255.0
            logging.debug(f"Type de données après normalisation: {image_normalized.dtype}")
            logging.debug(f"Min et Max de l'image normalisée: {image_normalized.min()}, {image_normalized.max()}")
            
            actions_mask = np.zeros_like(image_normalized)
            logging.debug(f"Forme de actions_mask: {actions_mask.shape}, Type: {actions_mask.dtype}")
            
            # Assurez-vous que l'image est suffisamment grande pour le slicing
            if image_normalized.shape[0] < 3 or image_normalized.shape[1] < 3:
                logging.error("L'image est trop petite pour effectuer le slicing requis.")
                return []
            
            # Appliquer les comparaisons pour détecter les "actions techniques"
            logging.debug("Application des comparaisons pour créer actions_mask.")
            actions_mask[1:-1, 1:-1] = (
                (image_normalized[1:-1, 1:-1] > image_normalized[:-2, 1:-1]) &
                (image_normalized[1:-1, 1:-1] > image_normalized[2:, 1:-1]) &
                (image_normalized[1:-1, 1:-1] > image_normalized[1:-1, :-2]) &
                (image_normalized[1:-1, 1:-1] > image_normalized[1:-1, 2:])
            )
            logging.debug("actions_mask calculé.")
            
            # Trouver les coordonnées des pixels détectés
            logging.debug("Recherche des actions détectées avec np.argwhere.")
            actions = np.argwhere(actions_mask)
            logging.debug(f"Nombre d'actions détectées avant compréhension de liste: {len(actions)}")
            
            # Convertir les coordonnées en tuples
            logging.debug("Conversion des coordonnées des actions en tuples.")
            actions = [(i, j) for i, j in actions]
            logging.debug(f"Nombre d'actions détectées après compréhension de liste: {len(actions)}")
            
            logging.debug(f"Actions techniques détectées: {len(actions)}")
            return actions
        except Exception as e:
            logging.error(f"detect_actions_techniques_in_image() - Exception: {e}")
            return []

    def estimateur(self, image_path: str) -> str:
        """
        Fonction principale pour analyser la posture dans une image donnée.
        Assure que la variable `result` contient toujours exactement 10 valeurs.
        Format: _val1_val2_val3_val4_val5_val6_val7_val8_val9_val10
        """
        # Initialisation des valeurs par défaut
        flexion_cou = 0
        flexion_cou_score = 0
        presence_personne = 0
        risk_zone = 0
        num_actions = 0
        repetitivite_score = 0
        maintien_posture_score = 0
        recuperation_score = 0
        prehension_score = 0
        dixieme_valeur = 0  # À modifier selon vos besoins

        # Liste pour collecter les valeurs
        fields = [
            flexion_cou, 
            flexion_cou_score, 
            presence_personne, 
            risk_zone, 
            num_actions,
            repetitivite_score, 
            maintien_posture_score, 
            recuperation_score, 
            prehension_score, 
            dixieme_valeur
        ]

        # Définir les seuils (réutiliser ceux de votre script initial)
        thresholds = {
            'flexion_cou': {
                'green': (-5, 20),
                'orange': [(20, 40)],
                'red': (40, float('inf'))
            },
            # Ajoutez d'autres seuils si nécessaire
        }

        try:
            logging.info(f"Analyse de l'image: {image_path}")
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"estimateur() - Impossible de charger l'image: {image_path}")
                return '_' + '_'.join(map(str, fields))
            logging.debug("Image chargée avec succès dans estimateur.")
            
            image_original = image.copy()
            image = self.preprocess_image(image)
            if image is None:
                logging.error("estimateur() - Prétraitement de l'image a échoué.")
                return '_' + '_'.join(map(str, fields))
            
            image_quality = self.calculate_image_quality(image)
            model_complexity = self.get_dynamic_model_complexity(image_quality)
            logging.info(f"Qualité de l'image: {image_quality}, Complexité du modèle: {model_complexity}")

            with self.mediapipe_lock:
                logging.info("Traitement de l'image avec Mediapipe Holistic...")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image_rgb)

            presence_personne = 1 if results.pose_landmarks else 0
            fields[2] = presence_personne  # Mise à jour de la valeur

            if presence_personne:
                landmarks = results.pose_landmarks.landmark
                keypoints = self.extract_keypoints(landmarks)

                required_keys = [KeyPoints.NECK_FLEXION, KeyPoints.SHOULDER_LEFT_ROTATION, KeyPoints.SHOULDER_RIGHT_ROTATION]
                if all(keypoints.get(key) is not None for key in required_keys):
                    flexion_cou = self.calculate_angle(
                        keypoints[KeyPoints.SHOULDER_LEFT_ROTATION], 
                        keypoints[KeyPoints.NECK_FLEXION], 
                        keypoints[KeyPoints.SHOULDER_RIGHT_ROTATION]
                    )
                    flexion_cou_score = self.classify_angle(flexion_cou, thresholds['flexion_cou'])

                    # Mise à jour des valeurs
                    fields[0] = flexion_cou
                    fields[1] = flexion_cou_score

                    # Détermination de la zone de risque
                    risk_zone = flexion_cou_score  # Simplification si les scores correspondent directement
                    fields[3] = risk_zone

                detected_actions = self.detect_actions_techniques_in_image(image_original)
                num_actions = len(detected_actions)
                fields[4] = num_actions

                # ***** Attribution des Scores Réels *****
                
                # 1. Repetitivite_score
                current_time = time.time()
                self.action_history.append((current_time, num_actions))
                
                # Nettoyage de l'historique pour ne conserver que les actions dans la fenêtre
                self.action_history = [
                    (t, a) for (t, a) in self.action_history if current_time - t <= self.repetitivite_window
                ]
                
                # Calcul de la répétitivité comme le nombre total d'actions dans la fenêtre
                repetitivite_score = sum(a for (t, a) in self.action_history)
                fields[5] = repetitivite_score
                logging.info(f"Repetitivite_score: {repetitivite_score}")

                # 2. Maintien_posture_score
                # Capture des positions clés pour le calcul de la posture
                posture = keypoints.copy()
                self.posture_history.append((current_time, posture))
                
                # Nettoyage de l'historique pour ne conserver que les postures dans la fenêtre
                self.posture_history = [
                    (t, p) for (t, p) in self.posture_history if current_time - t <= self.posture_window
                ]
                
                # Calcul de la variance des positions clés
                if len(self.posture_history) > 1:
                    variances = {}
                    for key in [KeyPoints.SHOULDER_LEFT_ROTATION, KeyPoints.SHOULDER_RIGHT_ROTATION, KeyPoints.ELBOW_LEFT, KeyPoints.ELBOW_RIGHT]:
                        positions = [p[key] for (t, p) in self.posture_history if p[key]]
                        if positions:
                            with self.keypoint_lock:
                                x_vals = [pos.x for pos in positions]
                                y_vals = [pos.y for pos in positions]
                            variance = np.var(x_vals) + np.var(y_vals)
                            variances[key] = variance
                    # Calcul de la variance moyenne
                    if variances:
                        avg_variance = np.mean(list(variances.values()))
                        # Inverser la variance pour que plus la variance est faible, plus le score est élevé
                        maintien_posture_score = max(0, 100 - avg_variance * 1000)  # Ajuster le facteur selon les besoins
                        fields[6] = round(maintien_posture_score, 2)
                        logging.info(f"Maintien_posture_score: {maintien_posture_score}")
                else:
                    fields[6] = 100  # Score maximal si aucune variation
                    logging.info("Maintien_posture_score: 100 (aucune variation)")

                # 3. Recuperation_score
                if self.last_activity_time:
                    elapsed = current_time - self.last_activity_time
                    if elapsed >= self.recuperation_threshold:
                        recuperation_score = min(100, elapsed)  # Score augmente avec le temps de récupération
                        fields[7] = recuperation_score
                        logging.info(f"Recuperation_score: {recuperation_score}")
                    else:
                        fields[7] = 0  # Pas encore de récupération suffisante
                        logging.info("Recuperation_score: 0 (récupération insuffisante)")
                else:
                    fields[7] = 0  # Aucune activité enregistrée
                    logging.info("Recuperation_score: 0 (aucune activité enregistrée)")

                # Mise à jour du dernier temps d'activité
                self.last_activity_time = current_time

                # 4. Prehension_score
                wrists = {
                    'wrist_left': keypoints.get(KeyPoints.WRIST_LEFT),
                    'wrist_right': keypoints.get(KeyPoints.WRIST_RIGHT)
                }
                self.hand_positions_history.append((current_time, wrists))
                
                # Nettoyage de l'historique pour ne conserver que les positions dans la fenêtre
                self.hand_positions_history = [
                    (t, p) for (t, p) in self.hand_positions_history if current_time - t <= self.prehension_window
                ]
                
                # Calcul de la stabilité des positions des poignets
                if len(self.hand_positions_history) > 1:
                    stability_scores = []
                    for wrist in ['wrist_left', 'wrist_right']:
                        positions = [p[wrist] for (t, p) in self.hand_positions_history if p[wrist]]
                        if positions:
                            with self.keypoint_lock:
                                x_vals = [pos.x for pos in positions]
                                y_vals = [pos.y for pos in positions]
                            variance = np.var(x_vals) + np.var(y_vals)
                            stability_scores.append(variance)
                    
                    if stability_scores:
                        avg_variance = np.mean(stability_scores)
                        # Inverser la variance pour que plus la variance est faible, plus le score est élevé
                        prehension_score = max(0, 100 - avg_variance * 1000)  # Ajuster le facteur selon les besoins
                        fields[8] = round(prehension_score, 2)
                        logging.info(f"Prehension_score: {prehension_score}")
                else:
                    fields[8] = 100  # Score maximal si aucune variation
                    logging.info("Prehension_score: 100 (aucune variation)")

                # ***** Ajustement Dynamique de la Complexité du Modèle *****
                self.adjust_model_complexity(fp=self.false_positives, fn=self.false_negatives)
        
        except Exception as e:
            logging.error(f"estimateur() - Exception: {e}")

        # Parcourir les champs et remplacer les valeurs manquantes par 0
        for i in range(len(fields)):
            if fields[i] is None:
                fields[i] = 0

        # Assurer que la liste contient exactement 10 éléments
        while len(fields) < 10:
            fields.append(0)

        # Si la liste contient plus de 10 éléments, tronquer les éléments supplémentaires
        fields = fields[:10]

        # Construction de la chaîne de résultat avec un underscore initial
        result = '_' + '_'.join(map(str, fields))

        logging.info(f"Résultat de l'analyse: {result}")
        return result

    # =============================================
    # Méthodes pour Gérer la Complexité du Modèle
    # =============================================

    def adjust_model_complexity(self, fp: int, fn: int) -> None:
        """
        Ajuste dynamiquement la complexité du modèle en fonction des performances récentes.
        Si trop de faux positifs ou faux négatifs sont détectés, réduit la complexité du modèle.
        """
        current_time = time.time()
        self.performance_history.append((current_time, fp, fn))
        
        # Nettoyage de l'historique des performances
        self.performance_history = [
            (t, fp_val, fn_val) for (t, fp_val, fn_val) in self.performance_history 
            if current_time - t <= self.performance_window
        ]

        # Évaluer les performances actuelles
        recent_errors = sum(fp_val + fn_val for (t, fp_val, fn_val) in self.performance_history)
        
        # Seuil pour ajuster la complexité
        if recent_errors > self.adjustment_threshold:
            self.current_model_complexity = 1
            logging.info("Ajustement de la complexité du modèle à 1 en raison des performances.")
        else:
            self.current_model_complexity = 2
            logging.info("Ajustement de la complexité du modèle à 2 pour une meilleure précision.")

    def get_dynamic_model_complexity(self, image_quality: float) -> int:
        """
        Détermine dynamiquement la complexité du modèle en fonction de la qualité de l'image
        et des performances récentes.
        """
        # Base sur la qualité de l'image
        model_complexity = 1 if image_quality < 0.5 else 2

        # Ajuster en fonction des performances récentes
        recent_errors = sum(fp + fn for (t, fp, fn) in self.performance_history)
        if recent_errors > self.adjustment_threshold:
            model_complexity = 1  # Réduire la complexité en cas de trop d'erreurs
            logging.info("Réduction de la complexité du modèle à 1 en raison des erreurs récentes.")
        else:
            model_complexity = 2  # Augmenter ou maintenir la complexité pour plus de précision
            logging.info("Augmentation ou maintien de la complexité du modèle à 2.")

        return model_complexity

    # =============================================
    # Fonction Périodique
    # =============================================

    def fct_periodique_1s(self) -> None:
        """Fonction périodique exécutée toutes les secondes pour gérer l'enregistrement et la communication."""
        retries = 0
        while not self.stop_event.is_set():
            try:
                HOST = self.full_ip_concentrateur
                PORT = 50000
                logging.info(f"Tentative de connexion au concentrateur {HOST}:{PORT}...")
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.settimeout(10)  # Timeout de connexion
                    client_socket.connect((HOST, PORT))
                    logging.info("Connecté au concentrateur.")
                    retries = 0  # Réinitialiser le compteur de tentatives

                    while not self.stop_event.is_set():
                        demande_recording = "yes"  # Cette logique peut être améliorée
                        if demande_recording == "yes":
                            self.recording = "yes"
                            filename = self.record_image()
                            if filename:
                                # Appeler la méthode estimateur avec le chemin de l'image
                                self.result_analyse = self.estimateur(filename)
                                try:
                                    os.remove(filename)  # Utilisation de os.remove au lieu de subprocess
                                    logging.info(f"Fichier {filename} supprimé.")
                                except Exception as e:
                                    logging.error(f"Erreur lors de la suppression de {filename}: {e}")
                                self.mdv_app()
                                now_message = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                                self.app_is_on = "yes"
                                message_emission = f"{self.num_poste}_{self.app_is_on}_{self.recording}_{self.pres_cam}_{self.mdv}_0{self.result_analyse}"
                                logging.info(f"{now_message} : {message_emission}")
                                try:
                                    client_socket.sendall(message_emission.encode())
                                    logging.info("Message envoyé au concentrateur.")
                                except (BrokenPipeError, ConnectionResetError) as e:
                                    logging.error(f"Erreur lors de l'envoi des données : {e}")
                                    break  # Sortir de la boucle pour tenter une reconnexion
                                except Exception as e:
                                    logging.error(f"Erreur inattendue lors de l'envoi des données : {e}")
                                    break
                        else:
                            self.recording = "no"
                        
                        time.sleep(1)  # Pause d'une seconde
            except (socket.timeout, socket.error) as e:
                logging.error(f"fct_periodique_1s() - Erreur de socket : {e}")
                retries += 1
                if retries > self.max_retries:
                    logging.error("Nombre maximal de tentatives de reconnexion atteint. Arrêt du thread périodique.")
                    break
                logging.info(f"Tentative de reconnexion dans {self.retry_delay} secondes...")
                time.sleep(self.retry_delay)
            except Exception as e:
                logging.error(f"fct_periodique_1s() - Exception : {e}")
                retries += 1
                if retries > self.max_retries:
                    logging.error("Nombre maximal de tentatives de reconnexion atteint. Arrêt du thread périodique.")
                    break
                logging.info(f"Tentative de reconnexion dans {self.retry_delay} secondes...")
                time.sleep(self.retry_delay)

    # =============================================
    # Méthodes pour Démarrer les Threads
    # =============================================

    def start_periodic_thread(self) -> None:
        """Démarre le thread périodique."""
        self.periodic_thread = threading.Thread(target=self.fct_periodique_1s, daemon=True)
        self.periodic_thread.start()
        logging.info("Thread périodique démarré.")

    # =============================================
    # Programme Principal
    # =============================================

    def run(self) -> None:
        """Exécute le programme principal de l'application."""
        # Configuration des arguments de la ligne de commande
        parser = argparse.ArgumentParser(description="Application de caméra avec versioning.")
        parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
        args = parser.parse_args()

        try:
            # Afficher la version au démarrage
            logging.info(f"Début de l'application CameraApplication, Version {__version__}")
            
            # Gestion des signaux pour un arrêt gracieux
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            logging.info(" Paramètres définis OK")
            
            # Vérification de la présence du concentrateur
            logging.info(f"Vérification de la présence du Concentrateur @{self.full_ip_concentrateur} ...")
            success = self.ping_host(host=self.full_ip_concentrateur)
            
            if success:
                logging.info(colored("... répond aux pings *", "green"))
            else:
                logging.warning(colored("... ne répond aux pings *", "red"))
            
            # Initialisation caméra ***********************************************
            logging.info("Initialisation AnywhereUSB...")
            self.init_anyusb()
            
            # Vérification si caméra connectée
            logging.info("Vérification de la connexion à la caméra...")
            cam_or_not = self.check_cam()
            if cam_or_not == 0:
                logging.warning("Caméra non détectée")
                self.pres_cam = "no"
                
                while self.check_cam() == 0 and not self.stop_event.is_set():
                    logging.info("Réessai de détection de la caméra dans 2 secondes...")
                    time.sleep(2)
                
                if self.stop_event.is_set():
                    logging.info("Arrêt demandé. Fermeture du programme.")
                    return  # Sortir si l'arrêt a été demandé
                
                try:
                    openni2.initialize()
                    logging.info("Connect_cam : OpenNI initialisé")
                    self.dev = openni2.Device.open_any()
                    logging.info("Connect_cam : caméra détectée")
                    self.color_stream = self.dev.create_color_stream()
                    self.color_stream.start()  # Démarrer le flux vidéo
                    logging.info("Prise en main de la caméra")
                    self.pres_cam = "yes"
                except Exception as e:
                    logging.error(f"run() - Exception lors de l'initialisation de la caméra : {e}")
                    self.pres_cam = "no"
            else:
                # Connexion à la caméra
                try:
                    openni2.initialize()
                    logging.info("Connect_cam : OpenNI initialisé")
                    self.dev = openni2.Device.open_any()
                    logging.info("Connect_cam : caméra détectée")
                    self.color_stream = self.dev.create_color_stream()
                    self.color_stream.start()  # Démarrer le flux vidéo
                    logging.info("Prise en main de la caméra")
                    self.pres_cam = "yes"
                except Exception as e:
                    logging.error(f"run() - Exception lors de la connexion à la caméra : {e}")
                    self.pres_cam = "no"
            
            if self.pres_cam == "yes":
                logging.info(colored("********** APPLICATION OPERATIONNELLE **********", "green"))
                logging.info(colored("Échanges en cours avec le concentrateur", "green"))
                
                # Démarrage de la fonction périodique dans un thread séparé
                self.start_periodic_thread()
                
                # **Suppression de l'affichage du flux vidéo**
                # self.start_video_thread()  # Cette ligne est supprimée
                
                # Maintenir le programme principal actif
                try:
                    while not self.stop_event.is_set():
                        time.sleep(10)
                except KeyboardInterrupt:
                    logging.info("Interrompu par l'utilisateur. Fermeture...")
                    self.sortie_programme()
                
                logging.info("FIN DE PROGRAMME")
                self.sortie_programme()
            else:
                logging.error("Caméra non initialisée. Arrêt du programme.")
                self.sortie_programme()
        except Exception as e:
            logging.error(f"run() - Exception générale: {e}")
            self.sortie_programme()

# =============================================
# Lancement de l'Application
# =============================================

if __name__ == "__main__":
    app = CameraApplication()
    app.run()

# =============================================
# FIN PROGRAMME PRINCIPAL
# =============================================
