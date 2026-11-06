# Plan : format BrainVision (.vhdr, .vmrk, .eeg) et une seule version de chaque outil

**Statut** : version 3 du 2026-11-06, confirmée, phase 1 lancée le même jour. Chaque phase est validée avant la suivante.
**Objectif immédiat** : pouvoir inspecter ces données et faire l'autoscoring (phases 1 à 3). Le reste de la chaîne suit
(phase 4), puis l'affichage haute densité (phase 5) et la suppression des jumeaux Curry (phase 6).
**Principe (ta remarque)** : les outils servent à d'autres jeux de données. Toute règle doit valoir pour n'importe quelle
étude, et les enregistrements mesurés ici servent seulement à la vérifier.
Ce fichier est versionné : il ne contient ni code participant, ni date d'enregistrement, ni nom de scoreur.

## Décisions prises (2026-11-06)

1. **Une seule version de chaque outil.** Les outils lisent `.edf` et `.vhdr` à travers un petit module de lecture, sans
   dossier de jumeaux. Les outils 0, 3bis et 5, qui n'ont pas de jumeau Curry, sont couverts du même coup.
2. **Les jumeaux Curry disparaissent** par la même méthode (phase 6). **Ils sont gelés de la phase 2 à la phase 6** : plus de
   régénération, pour ne pas adapter 166 lignes de générateurs qu'on jetterait ensuite. Une correction d'un outil EDF n'atteint
   donc pas Curry pendant ce temps : il ne faut pas laisser traîner la phase 6.
3. **Les fichiers compagnons sont renommés** : la détection des fichiers ne change pas (phase 0).
4. **La référence en ligne est recréée** avec la méthode de MNE (section *Référence en ligne*).
5. **Gain des canaux AUX : inféré et signalé, jamais corrigé automatiquement** (section *Canaux AUX*). Une correction sans code
   existe pour BrainVision.
6. **Triggers ignorés pour l'instant.** Leur gestion est en TODO dans le SPEC, après les ondes lentes et les fuseaux, avec les
   capteurs de pression des doigts.
7. **Qualité des canaux donnés aux modèles** : un contrôle général, tous formats, qui avertit et ne décide jamais à ta place
   (section *Qualité des canaux*).
8. **L'EOG sur une entrée EEG est re-référencé pour les modèles**, documenté et signalé (section *Référence de l'EOG*).
9. **IO est un EOG horizontal sous l'œil droit** (confirmé) : `eog_right` sur cette étude, un simple choix dans l'outil 2.
10. **TP9 et TP10 servent de mastoïdes** : ajoutés en fin de `LEFT_MASTOIDS` / `RIGHT_MASTOIDS` de l'outil 3bis, sans effet sur un
    EDF qui a M1/M2. Tout autre nom passe par le renommage de l'outil 2.
11. **`EMG` est bien un EMG du menton** (2 électrodes), il est donné aux modèles. Le dispositif de pression, ce sont les 5 canaux
    des doigts. **Ceci remplace la version 2**, dont le « pas un EMG du menton » était faux. Pour toutes les études, le nom reste
    un indice : la pré-sélection de l'outil 2 reste éditable, et l'outil 3bis écrit quel canal il utilise comme EMG du menton.
12. **La ceinture `Respi` est abdominale et donnée à SleepFM** : nouveaux alias dans `RESP_ALIASES['Abdominal']` (`Respi`, `Resp`,
    `Respiration`, `Resp Belt`, `Belt`). Sur une autre étude, ce rangement reste une convention, écrite dans le rapport. Aucun EDF local
    ni le jeu synthétique ne porte ces noms exacts (`Resp Imp` ne correspond pas), donc rien n'y change.
13. **Peu d'époques de référence dans l'outil 8bis** : sous 30 époques en N2/N3/R, tous les stades sont pré-cochés, avec un message
    visible. Le choix reste éditable, et une étude de nuit reste identique.
14. **Les noms de notebooks ne changent pas**, seuls leurs textes disent « recording ». Le renommage ira avec *Clean-up before
    distribution*.
15. **Hors périmètre** : l'outil 1bis (voir *Limites*) et les scripts batch de l'outil 1, qui restent EDF.

## Points qui restent ouverts

Aucun ne bloque les phases 1 et 2.

1. **Valeur exacte du gain** : à confirmer côté matériel quand tu auras l'info. L'inférence donne 100 en ordre de grandeur.
2. **Autre amplificateur** : tu n'as pas d'autre fichier. Ce qui vient du `[Comment]` reste donc traité comme optionnel, et c'est noté
   dans *Limites*.

## Mesures

Elles servent à vérifier les règles.

| Point | Résultat | Conséquence |
|---|---|---|
| Lecture MNE | `read_raw_brainvision` s'ouvre sans lire le signal (0,04 à 1,9 s) | MNE suffit pour l'en-tête, 4 avertissements à taire à chaque lecture |
| Montage | 2h45 à 3h16, 500 Hz, float32, 71 canaux. Entrées EEG : 62 EEG + IO, avec Cz en référence (tes 64 électrodes). AUX : 5 capteurs de pression (3 main gauche, 2 main droite), EMG du menton, ECG, ceinture | montage haute densité (phase 5) |
| Échelle EEG | valeur du fichier x 0,0488281 µV, écart-type médian 20 à 36 µV | correcte, identique à la résolution annoncée par le fabricant |
| Filtres | matériel DC à 140 Hz, logiciel 15,9155 s, soit 0,01 Hz (sauf Respi), moyenne médiane 7 à 8 µV | à lire dans le `[Comment]`, MNE range 0 Hz à cause de Respi |
| Référence | Cz, absente du fichier | décision 4 |
| Hypnogrammes | 3/3 avec exactement `floor(durée / 30)` lignes, labels W, 1, 2 | règle de longueur de l'outil 7 respectée |
| Événements | format texte Compumedics UTF-16, 205/205 sur la bonne époque depuis `New Segment`, stade identique à l'hypnogramme pour 205/205 | le parseur actuel convient, heure tronquée à la seconde comme en EDF |
| Pauses | 1 seul `New Segment` par fichier | une pause réelle décalerait l'heure : à détecter |
| Sommeil | 3 à 7 % des époques, pas de N3 ni de REM | à surveiller (phases 3 et 4) |
| Impédances | dans le `[Comment]`, `Out of Range!` sur 8 canaux d'un enregistrement | à montrer dans l'outil 1, sans s'y fier seul (voir *Qualité des canaux*) |
| IO | inutilisable sur 1 enregistrement sur 3 (impédance hors plage, environ 350 µV d'écart-type). Ailleurs, corrélé à F8-F7 (+0,82 sur le plus net), ce qui concorde avec ta confirmation | signalé par le contrôle |
| EMG | tonus plus fort en éveil qu'en N2 (361 → 277, 1 425 → 200, 665 → 158, à l'échelle du fichier), 95 % de la puissance à 50 Hz en N2 sur un enregistrement (4 époques) | vrai EMG, amplitude anormale, secteur à signaler |
| ECG | ondes R de 59 à 124 mV, rythme de 70 à 83/min, 54 à 67 % de la puissance à 50 Hz en N2 sur 2 enregistrements sur 3 | vrai ECG, amplitude anormale. SleepFM le reçoit sans filtre (comme Score_EDF) : à regarder en phase 3 |
| Ceinture | pic de 0,23 à 0,30 Hz (14 à 18/min), peu de dérive lente (0,2 à 0,3 fois la bande respiratoire) | utilisable par SleepFM |
| Capteurs de pression | puissance surtout sous 10 Hz | signaux de tâche, voir TODO des triggers |

**Trois pièges :**

- **TP9, TP10, FT9, FT10 ne sont pas dans `KNOWN_EEG_CHANNEL_RE`** : classés `other`, les mastoïdes seraient perdues. Le lecteur
  remplit `transducer_type` à partir du type MNE et de l'unité, voir *Architecture*. La regex ne change pas, donc les résultats EDF
  non plus.
- **Le CSV du PVT commence par le nom de l'enregistrement** (`<id>_S1_..._PVT.csv`) : il apparaîtra comme faux suffixe dans le menu
  *Event CSV* de l'outil 7. C'est sans effet sur la lecture, qui ne cherche qu'à côté de l'enregistrement. À vérifier en phase 4.
- **`Microéveil 5 MSE`, l'événement le plus fréquent, n'a pas de règle française** (`ARO SPONT` en a une) : il faudra le choisir à
  la main dans l'outil 4, y compris s'il doit ne rien signaler.

## Canaux AUX : inférer le gain

**Oui, on peut l'inférer, mais seulement en ordre de grandeur.**

| Mesure | Ces 3 enregistrements | 4 nuits EDF locales (référence) | Ce que ça dit |
|---|---|---|---|
| Pas de quantification de `Respi`, seul canal non filtré | multiples exacts de 0,298 µV (5 V / 2^24), écart de 0,001 contre 0,25 au hasard, sur 3/3 | sans objet | le fichier est bien en unités réelles au connecteur AUX, sans facteur 100 caché |
| Onde R de l'ECG, crête à crête | 124, 59, 72 mV | 0,31 à 0,56 mV | impossible au niveau de la peau : **le signal est amplifié avant le connecteur** |
| EMG, 10-100 Hz sans secteur, en N2 | 277, 200, 158 (éveil : 361, 1 425, 665) | 1,4 à 8,9 µV sur toute la nuit | environ 40 fois la référence |

- **Divisé par 100** : ECG de 1,2, 0,6 et 0,7 mV, EMG de 1,6 à 2,8 µV en N2 et de 3,6 à 14 µV en éveil. Tout redevient plausible, et un
  adaptateur BIP2AUX a justement un gain documenté de 100 (± 3 %).
- **Ce que le signal ne peut pas dire** : la valeur exacte. La physiologie varie trop pour séparer 50 de 100, seul le matériel le dira.
- **Limites du test ADC** : il ne vaut que pour ce matériel et pour ce canal, les autres étant filtrés. La fiche du fabricant donne
  10 V / 2^24 pour la révision 02 et je mesure 5 V / 2^24 : je n'explique pas ce facteur 2, mais ça ne change pas la conclusion.
- **Essayé, non retenu** : le plancher de bruit vers 105-135 Hz. EMG et ECG y sont 5 à 15 fois au-dessus des capteurs, mais rien ne
  permet d'en tirer un gain.

**Le contrôle général** (`channel_check_lib.context_amplitude_warnings`, tous formats)

Il repose sur la physiologie, donc il marche sans rien savoir du matériel. Il tourne là où ces canaux sont lus : outils 3bis et 7.

| Canal déclaré | Plage attendue | Hors plage |
|---|---|---|
| ECG | onde R de 0,1 à 5 mV, rythme de 30 à 120/min | ⚠ avec le facteur estimé (onde R / 1 mV, arrondi à la décennie), ou « does not look like an ECG » si le rythme est hors plage |
| EMG du menton | RMS 10-100 Hz sans secteur, jusqu'à 100 µV | ⚠ avec le facteur estimé (RMS / 5 µV, arrondi à la décennie) |
| ECG et EMG | | ⚠ si le secteur (50 ou 60 Hz) domine |

- **Essai** : sur 4 nuits EDF (4 ECG, 16 EMG), aucune alerte. Sur ces 3 enregistrements, l'ECG et l'EMG tombent tous les deux sur la
  décennie 100. Plages à recaler en phase 3, y compris sur Curry : 4 nuits d'un seul système, c'est peu.
- **Le message dit bien « ordre de grandeur, à confirmer »** et ne corrige rien.

**Effet sur l'autoscoring** (lu dans le code) : GSSC (pas d'EMG, signaux z-scorés) et SleepFM (signaux standardisés) ne dépendent pas
de l'échelle. **YASA en dépend pour son EMG** (`std` et `iqr` en µV absolus). L'outil 8 aussi pour l'affichage (échelles fixes).

**Corriger.** Pour BrainVision, sans code : changer la résolution dans le `.vhdr` (`Ch69=EMG,,0.01,mV`, `Ch70=ECG,,0.01,mV`), en gardant
l'original à côté. Le signal n'est pas touché et tous les outils voient la bonne échelle. Testé sur une copie temporaire : rapport de 100
sur ces canaux, de 1 sur C3. Les capteurs de pression et la ceinture ne sont pas concernés.

**Sur cette étude, tant que le gain n'est pas connu (décidé)** : YASA avec `None` pour son EMG. La phase 3 compare trois runs de YASA au
scoring expert (sans EMG, EMG tel quel, EMG corrigé sur une copie) : ça dira ce que le choix change vraiment.

**Pour les autres formats** : pas de correction simple. Le message propose YASA sans EMG. Un facteur par canal dans l'outil 2 reste
possible (additif, absent par défaut), je ne l'ajoute que si un cas réel se présente.

**Outil 1** : il ne lit pas le signal. Il affiche seulement une ligne d'information pour un ECG ou un EMG en AUX avec un gradient de
1 mV/mV, et renvoie au contrôle.

## Qualité des canaux donnés aux modèles

**Le risque.** Une mauvaise référence peut être contre-productive. `C3-M2` vaut `C3 - M2` : tout le bruit de M2 s'ajoute à chaque
dérivation qui l'utilise.

| Situation | Effet |
|---|---|
| YASA (une seule dérivation) | touché directement, et ses mesures d'amplitude sont absolues |
| GSSC en consensus | C3 et F3 passent par M2, C4 et F4 par M1 : un seul côté touché, donc protection partielle, mais le vote le plus confiant n'est pas forcément le bon |
| SleepFM (signaux standardisés) | le bruit gonfle l'écart-type et écrase le vrai signal |
| Mastoïde morte | la dérivation garde la référence en ligne : mauvaise référence, sans aucun signe visible |
| Bruit à 50 Hz seul | peu grave sur l'EEG, que les modèles filtrent en dessous |

Une électrode de scoring mauvaise pose le même problème, et l'impédance ne suffit pas : sur un enregistrement sans aucune impédance
hors plage, C3 et 9 autres canaux sont très bruités. **Le contrôle porte donc sur tous les canaux que les modèles reçoivent.**

**Le contrôle** (`channel_check_lib.channel_quality_warnings`, tous formats)

- **Méthode** : 60 époques de 30 s réparties sur l'enregistrement, filtrées de 0,3 à 35 Hz (ce que les modèles voient). Chaque canal
  est comparé au canal médian de la même époque, parmi les canaux EEG de la configuration.
- **Électrode ou mastoïde** : *bruitée* si son écart-type dépasse 3 fois la médiane sur plus de 20 % des époques, *morte* s'il passe
  sous 0,1 fois la médiane sur plus de 20 % des époques.
- **EOG** : les mouvements des yeux le rendent incomparable à l'EEG. Il est signalé si sa médiane dépasse 100 µV, ou s'il est plat.
- **Le résultat est un avertissement.** L'enregistrement est scoré quand même et rien n'est changé automatiquement : changer la
  référence d'un participant sans le dire rendrait les scorings incomparables entre eux.
- **Où on le voit** : ⚠ dans le log et le rapport, et clé additive `channel_quality_warnings` dans le fichier de paramètres,
  absente s'il n'y a rien à dire. Les sorties restent donc identiques sans avertissement.
- **Ce que le message propose**, avec ce qui existe déjà : l'électrode de l'autre côté (C4-M1 si M2 est mauvaise), `EEG reference =
  As declared in tool 2`, `EOG = None`, et le sélecteur de participants pour relancer celui-là seul.

**Essai du critère** (seuils à caler en phase 3)

| Données | Résultat |
|---|---|
| 13 EDF locaux (4 à 12 EEG, vraies nuits et défauts injectés) | le canal mort du jeu de test est trouvé sur 100 % des époques, une seule autre alerte (Fp2 sur `100`, 47 %, à regarder) |
| Défauts non détectés sur ces EDF | clipping, 50 Hz, dérive, quantification, coupure de 30 min, mouvements brefs : le critère vise le bruit durable et les canaux morts. L'outil 6 reste là pour le reste |
| 3 enregistrements BrainVision, 62 canaux | les 7 canaux EEG à impédance hors plage sont tous trouvés, avec d'autres dont l'impédance était bonne (dont C3 sur un enregistrement). Mastoïdes non signalées (TP9 à 18 % sur l'un). Pas de faux canal mort |
| Mêmes enregistrements, petit montage de 8 canaux | mêmes conclusions sur ces 8 canaux (O2 sur le premier, C3 sur le deuxième, rien sur le troisième) |
| Électrodes voisines de la référence en ligne | rapport médian le plus bas de 0,26. C'est pourquoi je propose 0,1 pour *morte* plutôt que le 0,2 de l'essai, à vérifier en phase 3 sur le canal mort |

**Calage en phase 3** : EDF de test (avec `test_data_manifest.tsv`), Curry, BrainVision, et EOG d'autres études pour le seuil de 100 µV.

**Même risque dans l'outil 7** : une référence déclarée mauvaise contamine tous les canaux. Le même contrôle s'applique à ses canaux
de référence en phase 4, sous forme de ⚠ dans le log seulement, donc aucune sortie ne change.

## Référence de l'EOG

**Ce que ça veut dire.** Dans le fichier, IO vaut `œil - Cz`, car toutes les entrées EEG sont enregistrées par rapport à la
référence en ligne. Ce signal contient donc aussi l'EEG du vertex, inversé : fuseaux, complexes K, ondes lentes, très forts à Cz. Les
modèles ont appris sur `œil - mastoïde` (E1-M2), où la mastoïde apporte peu d'EEG. Soustraire la mastoïde fait disparaître Cz :
`(œil - Cz) - (M - Cz) = œil - M`.

**La règle, générale :**

| Cas | Ce qui est fait |
|---|---|
| EOG sur une entrée EEG (unité µV, référence en ligne connue) | même règle controlatérale que l'EEG : `E1-M2`, `E2-M1` |
| EOG sur une entrée AUX | déjà bipolaire, intact |
| Deux EOG en mode `L − R bipolar` | intact, le terme commun s'annule |
| `EEG reference = As declared in tool 2` | même référence que l'EEG |
| Mastoïde absente | EOG tel quel, avec un ⚠ |
| EDF | inchangé : le fichier ne dit pas quelle est la référence, et les exports sont en général déjà référencés |
| Curry | à vérifier en phase 6, inchangé d'ici là |

- **Documentation** : la dérivation réelle (ici `EOG-R = IO-M1`) est écrite dans le fichier de paramètres et le rapport, et l'EOG passe
  par le contrôle de qualité avec sa mastoïde.

## Référence en ligne

`mne.add_reference_channels(raw, [nom], copy=False)` est la méthode documentée par MNE (tutoriel *Setting the EEG reference*), et
réintégrer la référence en ligne avant une référence moyenne est la pratique courante. Testé avec Cz : le canal est nul et de type EEG
après l'ajout, et vaut exactement `-(TP9+TP10)/2` après référence aux mastoïdes.

- **N'importe quel nom** : `Reference Channel Name` du `[Comment]`, qui peut être FCz sur un autre montage.
- **Condition** : ajouté seulement si le `[Comment]` donne ce nom et que ce canal manque dans le fichier. Sinon rien n'est ajouté, et
  l'outil 1 dit que la référence n'est pas indiquée.
- **Position** : MNE la laisse à NaN. Le lecteur la prend dans un montage standard de MNE à la même échelle de tête, sinon les
  topographies de l'outil 8 tomberaient.
- **Outils 1 et 2** : une ligne indique « online reference, not in the file ».
- **Outil 6** : il ne re-référence pas, donc ce canal serait plat et faussement signalé. Il est laissé de côté, avec une ligne qui le
  dit.
- **Sans re-référence** : le canal reste plat. L'outil 2 avertit s'il est choisi sans référence.
- **À savoir** : la référence moyenne hérite des mauvais canaux. Sur la fenêtre du test, où des canaux sont très artefactés (jusqu'à
  3 790 µV d'écart-type), Cz a 90 µV d'écart-type, contre 20 µV avec les mastoïdes.

## Architecture

**Nouveaux fichiers et ajouts**

- `tools/brainvision_lib.py` : l'en-tête au format de `read_edf_header_custom`, `read_vmrk_info` (heure de début, nombre de segments),
  `read_comment_info` (référence, filtres, impédances, gradients), la lecture du signal. Équivalent de `curry_header.py` + `curry_io.py`.
- `tools/recording_lib.py` : `read_header(path)`, `read_start_datetime(path)`, `read_raw(path, include=None, add_online_reference=True)`.
  Un simple `if` sur l'extension. La branche EDF appelle exactement le code actuel.
- `tools/channel_check_lib.py` : `channel_quality_warnings`, `context_amplitude_warnings`. Sans torch, importable par les outils 0, 3bis, 7.
- `study_files_lib.py` : `RECORDING_SUFFIXES`, `list_recordings(folder)`. Il reste sans lecture de signal.
- Rien à ajouter à `environment.yml` : MNE lit ce format sans dépendance.

**Pas de parseur binaire maison.** Pour l'en-tête, MNE gère déjà `INT_16`, `VECTORIZED`, la virgule codée `\1` et les codepages, ce qui est
plus sûr sur d'autres jeux de données. Seuls le `[Comment]` et le `.vmrk` sont lus comme du texte.

**Ce que fournit chaque format**

| Besoin | EDF (inchangé) | BrainVision |
|---|---|---|
| Trouver | `.edf` | `.vhdr`, jamais `.eeg` (extension aussi utilisée par Nihon Kohden) |
| En-tête | `read_edf_header_custom` | même dictionnaire, `physical_*` et `digital_*` vides, `transducer_type` calculé |
| Fréquence, durée | champs de l'en-tête | MNE (`sfreq`, `n_times`) |
| Type de canal | transducteur ou nom | `EOG`, `EMG`, `ECG` si le nom le dit (IO, SO, LO, HEOG, VEOG inclus) ; sinon `EEG` si MNE le type EEG et que l'unité est en µV ; sinon vide |
| Heure de début | octets 168 et 176 | `New Segment` du `.vmrk`, tronqué à la seconde |
| Signal | `read_raw_edf(include=)`, suffixes `-0` | `read_raw_brainvision`, `pick`, référence en ligne, annotations retirées (les `.fif` ne portent pas les triggers) |
| Bornes physiques | oui | aucune : `bounds_pct` absent, comme en Curry |
| Plafond d'affichage | non | ±500 µV |
| High-pass de l'outil 6 coché par défaut | non | seulement si l'acquisition n'a aucun high-pass (DC), comme le jumeau Curry |

**La règle se déduit des données et pas du format** : pas de bornes physiques = pas de `bounds_pct` et plafond ±500 µV ; acquisition
DC = high-pass coché. Curry en profitera sans rien de spécifique. Pour ce high-pass, le `display` initial suit la case (règle
*Checkbox-revealed widgets*).

**Cas à traiter dans le lecteur**

- **Le `[Comment]` n'a pas de format garanti** et je ne l'ai vu que pour un amplificateur : référence, filtres, impédances et gradients
  sont tous optionnels. Absents ou illisibles, ils ne sont pas affichés et rien ne s'arrête ;
- `.eeg` absent ou nommé autrement que dans `DataFile` : échec clair, l'enregistrement passe dans `failed` ;
- plusieurs `New Segment` : ⚠ dans l'outil 1, et l'heure n'est plus fiable, donc l'export texte des événements n'est pas converti ;
- `New Segment` sans date : pas d'heure de début, comme un en-tête EDF illisible ;
- même nom en `.edf` et en `.vhdr` : ⚠ au scan, un seul est gardé.

## Phases

Chaque phase est validée avant la suivante, avec des commits séparés. Je propose un niveau de tests à chaque fois et j'attends ta
réponse. Dans toutes les phases, **le golden EDF synthétique ne doit pas bouger**.

### Phase 0 : les fichiers (toi, ou moi avec ton accord car c'est hors du projet)

| Avant | Après |
|---|---|
| `Hypnogram_<id>_<scoreur>.txt` | `<id>_Hypnogram_<scoreur>.txt` |
| `PSG4_ScoredEvents_<id>_Export<scoreur>.txt` | `<id>_ScoredEvents_Export<scoreur>.txt` |

- **Ne jamais renommer `.vhdr`, `.vmrk`, `.eeg`** : le `.vhdr` contient les noms des deux autres.
- La règle générale, à écrire dans le SPEC et le README : un compagnon s'appelle `<nom de l'enregistrement><suffixe>`.
- Mes tests travaillent sur une copie dans `%TEMP%\bvt` (4,4 Go, 176 Go libres), supprimée à la fin. Ton dossier n'est pas touché.
- **Pour tes propres runs** : les outils écrivent dans le dossier d'étude, et un `.fif` de l'outil 7 pèsera environ 1,4 Go par
  enregistrement. Mieux vaut un dossier d'étude hors OneDrive.

### Phase 1 : le lecteur, sans toucher aux outils

- `brainvision_lib.py`, `recording_lib.py`, `list_recordings`.
- `tests/test_brainvision_lib.py`, avec des enregistrements écrits à la main : `INT_16` et float32, multiplexé et vectorisé, codepages,
  `[Comment]` absent ou tronqué, autre référence que Cz, et le signal relu égal au signal écrit.
- **Tests proposés** : quick + ces nouveaux tests.

### Phase 2 : inspection (outils 1, 2, 3, 6)

- **Outil 1** : table des canaux, résolution, filtres du `[Comment]`, impédances avec seuil en kΩ (20 par défaut, comme le jumeau Curry)
  et `Out of Range!` signalé, pauses, ligne sur le gain AUX. L'anonymisation et la plage dynamique affichent « not applicable ».
- **Outil 2** : scan par `read_header`, référence en ligne, détection du contexte, aperçu par `read_raw`.
- **Outil 3** : découverte des enregistrements seulement.
- **Outil 6** : bloc de lecture, règles des bornes physiques et du high-pass, référence en ligne laissée de côté.
- **Tests** : `tests/synthetic.py` écrit aussi un jeu BrainVision. Test d'équivalence : le même signal en EDF et en BrainVision donne
  les mêmes métriques dans l'outil 6, à la tolérance près. Puis lancement sur les vraies données en `%TEMP%`.
- **Tests proposés** : standard sur `tool1`, `tool2`, `tool3`, `tool6`.

### Phase 3 : autoscoring (outil 3bis). **Objectif atteint, arrêt pour ton test**

- `autoscoring_lib.py` : les 5 `read_raw_edf` passent par `read_raw`, le scan du notebook par `read_header`.
- Référence de l'EOG, dans les trois chemins : GSSC (`load_eog`), YASA, SleepFM.
- Mastoïdes TP9 et TP10, retirées des électrodes proposées au scoring (`eeg_options`).
- Alias de la ceinture pour SleepFM.
- `channel_check_lib.py` : les deux contrôles, calage des seuils sur EDF, Curry et BrainVision, test sur les défauts du jeu de test.
- Le choix par défaut des électrodes ne dépend pas de la densité (GSSC : C3, C4, F3, F4, YASA : C4) : rien à changer.
- Section 4 sur les vraies données, contre ton scoring expert.
- **Comparaisons sur ces données** : YASA sans EMG, avec EMG tel quel et avec EMG corrigé ; SleepFM avec et sans ECG (50 Hz), avec et sans
  ceinture.
- **Fiabilité à juger avec la Section 4** : les modèles ont appris sur des nuits entières, ici 93 à 97 % d'éveil pendant une tâche. C'est
  une question scientifique que le format ne règle pas.
- **Tests proposés** : standard sur `tool3bis`, puis `tests/score_edf_equivalence.py`, car la Section 4 ne doit pas changer.
- **Fin de phase** : tu testes dans Voila.

### Phase 4 : le reste de la chaîne (outils 4, 5, 7, 0)

- **Outil 4** : découverte, événements, heure de début.
- **Outil 5** : `read_edf_header_info` et les lectures `include=[canal]` passent par le lecteur.
- **Outil 7** : les 5 sites de découverte, les blocs `[load]` et `[H]`, l'heure de début, la référence de l'EOG dans le compagnon, les ⚠ sur
  les canaux de référence et sur l'amplitude des canaux de contexte.
- **Outil 0** : par `edf_io_lib`, et les contrôles à son étape 3.
- **Vérifier sans modifier** : 5bis, 8, 9 sur des sorties BrainVision.
- **Outil 8bis** : pré-cochage de tous les stades sous 30 époques de référence (`default_reference_stages`), avec un test qui prouve qu'une
  étude de nuit ne change pas.
- **Points à regarder** :
  - les métriques de l'outil 5 sur des siestes en journée, et ses *lights off / on* ;
  - la mémoire de l'outil 7 : environ 2,9 Go par enregistrement avant ses copies, sans problème sur 31 Go.
- **Tests proposés** : full, une fois.

### Phase 5 : affichage haute densité

Le principe reste : **adapté au nombre de canaux, jamais au format, inerte à 12 canaux ou moins** (`HD_CHANNEL_THRESHOLD`), donc EDF
inchangé. Audit à 62 canaux :

| Outil | État lu dans le code | À faire |
|---|---|---|
| 8, 8bis | adaptés dans `qc_rejected_epochs_lib` | vérifier à 62 canaux (Curry allait jusqu'à 44) |
| 9 | **une rangée de n colonnes, largeur plafonnée à 20 pouces** (`_fig_width`), soit 0,3 pouce par canal | à refaire (grille ou sélection de canaux), pas d'usage connu en haute densité |
| 6 | grilles `5 x ncols` par `3,5 x nrows`, superposition déjà transparente | lisibilité et poids du rapport HTML |
| 7 | heatmap et rapport | vérifier |
| 1, 2 | tables et listes de 71 canaux | vérifier |
| 0 | navigateur et explorateur | vérifier, pas d'usage connu en haute densité |
| 3bis | cases à cocher sur plusieurs lignes, choix par défaut indépendant de la densité | vérifier l'affichage |
| 5 | pas de signal EEG | rien |

- **Tests proposés** : quick, plus un test qui prouve que rien ne change à 12 canaux ou moins.

### Phase 6 : suppression des jumeaux Curry

**Faisable, et je le recommande.** D'après le SPEC, ce que font les jumeaux tient en peu de choses :

| Outil | Ce que fait le jumeau | Dans la version unique |
|---|---|---|
| 1 | page dédiée, contrôle des impédances | déjà là en phase 2 ; sorties dans `reports_inspection/` et plus `reports_inspection_curry/` |
| 2 | masque EEG par groupe du `.dpo` | `transducer_type` rempli depuis ce groupe |
| 3 | découverte | rien de plus |
| 4 | une seule source d'événements, pas de contrôle croisé | même chaîne TXT, contrôle montré seulement avec deux sources |
| 6 | sans `bounds_pct`, plafond ±500 µV, high-pass coché | déjà les règles de phase 2 |
| 7 | lecture Curry, heure du `.dpo` | par le lecteur |
| 8, 8bis | titre et chemin d'import seulement | jumeaux supprimés, rien à porter |

**Ce qui en découle**

- `curry_header.py` et `curry_io.py` deviennent `tools/curry_lib.py`, branché dans `recording_lib.py`.
- Les outils 0, 3bis, 5 fonctionnent aussi sur Curry. Deux TODO se referment : *A Curry twin of tool 0* et *EDF reading of tools 2, 4, 6, 7
  through `edf_io_lib`*. Plus aucun générateur ne réécrit ces blocs : les copies peuvent disparaître et `test_edf_io_sync.py` se réduit.
- **Supprimés** : 8 notebooks, 7 générateurs, `test_curry_twins.py`. `test_notebooks_execute.py` et `test_ui_lib.py` sont adaptés.
- `tools_curry/inspect_curry.py` n'est pas versionné : je te le demande avant d'y toucher.

**Validation** : c'est la seule phase où l'on peut perdre quelque chose, mais les 2 enregistrements de `tools/test_data_curry/` (7 Go
chacun) permettent de la vérifier.

1. Faire tourner les jumeaux actuels et sauvegarder toutes les sorties, avant toute suppression.
2. Faire tourner les outils uniques sur les mêmes données.
3. Comparer tous les fichiers (`tests/refactor_check.py`) : identiques, sauf différences listées et justifiées. Run long, avec keep-awake.
4. Tu vérifies dans Voila sur Curry, comme le SPEC le demandait déjà.

- **Tests proposés** : full, plus cette comparaison.

### Phase 7 : documentation et fin

- **SPEC** : section *Recording formats* (EDF, BrainVision, Curry) à la place de *Curry 9 support*, structure du dépôt, les deux contrôles,
  référence de l'EOG, alias de respiration (avec le TODO *SleepFM's respiration* de §3bis), TODO fermés, ordre des TODO.
- **CLAUDE.md** : la partie *Curry twins* se réduit fortement, et une règle courte s'ajoute (toute lecture passe par `recording_lib`).
- **README et TESTS.md** à jour.
- Ce plan est supprimé en commit séparé.
- Les modifications du SPEC te sont proposées à chaque phase.

## Limites connues

- **Autres amplificateurs BrainVision** : la lecture passe par MNE, mais le `[Comment]` n'a été vu que pour l'actiCHamp, et je n'ai pas
  d'autre fichier pour le vérifier.
- **Outil 1bis** : il reste EDF, car le `.vhdr` n'a pas de champ patient. Mais le `.vmrk` porte la date d'enregistrement et le `[Comment]` le
  numéro de série de l'ampli : avant tout partage de données, il faudra une anonymisation dédiée. À noter en TODO si tu le souhaites.
- **Données compressées ou segmentées** (exports Analyzer) : non couvertes, seul l'enregistrement continu l'est.
- **Triggers et capteurs de pression** : lus mais inutilisés, voir SPEC *Task triggers*.
- **Outil 6** : ses drapeaux ne repèrent pas un canal simplement bruité (ils visent les canaux plats, morts, écrêtés ou bimodaux). Y ajouter
  le critère du contrôle changerait les résultats EDF : ce sera un TODO à part si tu le souhaites.

## Sources sur le matériel

- [BIP2AUX Adapter, gain et fonction](https://www.brainproducts.com/?p=11597)
- [Sensor Tutorial, BIP2AUX Adapter](https://www.brainproducts.com/files/public/sensor-tutorial/Content/Topics/3.BIP2AUXAdapter/BIP2AUXAdapter.htm)
- [actiCHamp, plages d'entrée et résolutions](https://www.brainproducts.com/solutions/actichamp/)
