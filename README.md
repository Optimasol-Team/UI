Tout d’abord, téléchargez depuis le dépôt de l’équipe les dossiers Optimiser_Engine-v2.0 et application_v1, puis décompressez-les et placez-les dans le dossier UI (en raison des limitations de taille de fichiers sur GitHub, ils ne sont pas inclus directement dans ce dépôt).
Vérifiez que l’arborescence est correcte, en particulier que index.html se trouve bien dans le dossier static.

Ensuite, créez un environnement virtuel :
python -m venv venv

Activez-le :
venv\Scripts\activate

Installez les dépendances :
pip install -r requirements.txt

Puis effectuez l’installation en mode éditable du moteur d’optimisation. Placez-vous dans son dossier :
cd Optimiser_Engine-v2.0-main
pip install -e .

Revenez ensuite au dossier racine du projet :
cd ..

Lancez l’application backend avec :
uvicorn app1:app --reload

Pour arrêter le serveur, utilisez Ctrl + C dans le terminal afin de libérer le port.

⚠️ Important : l’installation en mode éditable (pip install -e .) est indispensable, sinon des erreurs d’import apparaîtront. Cette étape ne peut pas être incluse dans le fichier requirements.txt et doit être exécutée manuellement.
