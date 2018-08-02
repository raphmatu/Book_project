# Book_project
Judging a book on its cover


Tutorial pour télécharger les images :

- Récupérer le fichier "Download_images.py"
- Récupérer les fichiers "book30-listing-train.csv", "book30-listing-test.csv" et "book32-listing.csv"
- Attention : Assurez vous que ces 4 fichiers soient dans le même dossier !
- Ouvrir Anaconda
- Ouvrir une page JupyterLab
- Ouvrez un terminal Bash (Launcher ==> Other ==> Terminal)
- A l'aide de lignes de commande, allez dans le dossier où se trouvent les 4 fichiers cités plus haut
- Une fois dans le dossier, taper la commande "python .\download_images.py .\Database_train .\book30-listing-train.csv"
- Le téléchargement devrait se lancer et stocker les images dans un dossier

Remarques importantes :

- Vous devez installer les packages tqdm et joblib grâce à l'environnement anaconda pour que le fichier python fonctionne
- Attention à ne pas télécharger les photos dans un dossier que vous synchroniseriez avec Git :p
