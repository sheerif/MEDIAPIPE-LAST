#!/bin/sh

#fichier PID
pid_file_recording="/home/pc-camera/Bureau/Cameras/03_Code_MiniPC/mastersys_recording.pid"
pid_recording=""
CHEMIN="/home/pc-camera/Bureau/Cameras/03_Code_MiniPC/"
LOGFILE="${CHEMIN}/mastersys_recording.log"

start_programme() {
    cd "$CHEMIN"

    # Démarrer le prog
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Démarrage du prog recording.py ..." >> $LOGFILE 2>&1
    python3 recording.py & 
    # Enregistrer le PID dans le fichier
    pid_recording=$!
    echo "$pid_recording" > "$pid_file_recording"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - pid_recording : $pid_recording" >> $LOGFILE 2>&1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - pid_file_recording : $pid_file_recording" >> $LOGFILE 2>&1
}

stop_programme() {

    # Récupérer les PID depuis le fichier
    pid_recording=$(cat $pid_file_recording)
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $pid_recording" >> $LOGFILE 2>&1

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Arrêt  avec le PID $pid_recording..." >> $LOGFILE 2>&1
    kill -9 $pid_recording  >> $LOGFILE 2>&1

    #arrêt du pid en cours
    #kill -9 $$

    # Supprimer le fichier PID
    rm -f "$pid_file_recording"  >> $LOGFILE 2>&1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - programme arrêté." >> $LOGFILE 2>&1
}


#-----------------------------------------------------#
#                       MAIN                          #
#-----------------------------------------------------#

# Exécuter l'action spécifiée par l'utilisateur
case "$1" in
    start)
        # Vérifier si en cours d'exécution
       if pgrep -x "recording.py"; then
           echo "$(date '+%Y-%m-%d %H:%M:%S') - recording est  en cours d'exécution." >> $LOGFILE 2>&1
           exit 1
        fi
        
        # Supprimer le fichier PID
        rm -f "$pid_file_recording"  >> $LOGFILE 2>&1
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ########################################   1er START PROGRAMME   ########################################" >> $LOGFILE 2>&1
        #premier start
        start_programme
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ######################################## FIN 1er START PROGRAMME ########################################" >> $LOGFILE 2>&1
       
        # Surveiller l'existence du fichier PID et relancer le programme si nécessaire
        pid_recording=$(cat $pid_file_recording)
        echo "$(date '+%Y-%m-%d %H:%M:%S') - pid_recording : $pid_recording" >> $LOGFILE 2>&1
        echo "$(date '+%Y-%m-%d %H:%M:%S') - pid_file_recording : $pid_file_recording" >> $LOGFILE 2>&1
    	  while true; do
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ##### DANS LA BOUCLE 30s d'attente..." >> $LOGFILE 2>&1
            sleep 30  # Attendre x secondes avant de vérifier à nouveau
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ##### DANS LA BOUCLE => TEST DE L'EXISTANCE du PID pid_recording : $pid_recording" >> $LOGFILE 2>&1
            if kill -0 $pid_recording > /dev/null 2>&1; then
               echo "$(date '+%Y-%m-%d %H:%M:%S') - Le processus avec PID $pid_recording est en cours d'exécution" >> $LOGFILE 2>&1
            else
               echo "$(date '+%Y-%m-%d %H:%M:%S') - Le programme recording.py est arrêté ##### RE-START PROGRAMME" >> $LOGFILE 2>&1
               start_programme
               echo "$(date '+%Y-%m-%d %H:%M:%S') - ##### FIN DU RE-START PROGRAMME" >> $LOGFILE 2>&1
               #break
            fi
        done
        ;;
    stop)
        stop_programme
        ;;
    *)
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Action non reconnue. Utilisation: $0 {start|stop}" >> $LOGFILE 2>&1
        exit 1
        ;;
esac
