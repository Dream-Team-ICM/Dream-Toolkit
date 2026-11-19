# Plan : format BrainVision (.vhdr, .vmrk, .eeg) et une seule version de chaque outil

**Statut** : version 3 du 2026-11-06, confirmée. Phases 1 à 5, 4 ter, 5 bis et 5 ter faites et commitées (5 ter avec la
correction A + B de l'outil 0) et 5 quater. Phase 6 (suppression des jumeaux Curry) faite le 2026-11-19 (tests quick, standard et full passés).
Les phases 6 bis (bouton Stop) et 6 ter (numérotation des dossiers de rapports) passent dans un même TODO du SPEC le 2026-11-19
(*Voila pages: a Stop button and numbered report folders*). Reste la phase 7.
Chaque phase est validée avant la suivante.
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
| Impédances | dans le `[Comment]`, un peu avant le début. `Out of Range!` sur 8 canaux d'un enregistrement, `not connected` sur 9 d'un autre, rien sur le 3e. Tous ces canaux EEG sont très bruités (écart-type de 3 400 à 18 000 µV) | à montrer dans l'outil 1 tel que le fichier l'écrit, sans s'y fier seul (voir *Qualité des canaux*) |
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

**Confirmé le 2026-11-12** : l'EMG et l'ECG passent par un adaptateur BIP2AUX (gain 100) avant l'entrée AUX, et le workspace laisse
le gradient à 1 mV/mV. Le facteur 100 vient donc du matériel, non compensé à l'enregistrement. La ceinture et les capteurs de
pression ne passent pas par l'adaptateur.

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
possible (additif, absent par défaut), je ne l'ajoute que si un cas réel se présente. **Le cas réel est là** (BIP2AUX confirmé) : voir
la phase 4 ter, que tu préfères à la modification des `.vhdr`.

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

Une électrode de scoring mauvaise pose le même problème, et l'impédance ne suffit pas : sur l'enregistrement dont toutes les impédances
sont bonnes, 7 canaux sont très bruités pendant l'enregistrement, dont TP9 (corrigé le 2026-11-06 : j'avais d'abord cité C3 sur un autre
enregistrement, où il est en réalité `not connected`). **Le contrôle porte donc sur tous les canaux que les modèles reçoivent.**

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
| 3 enregistrements BrainVision, 62 canaux | les canaux `Out of Range!` et `not connected` sont trouvés, avec d'autres dont l'impédance était bonne. Mastoïdes non signalées (TP9 à 18 % sur l'un). Pas de faux canal mort |
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
| Type de canal | transducteur ou nom | `EOG`, `EMG`, `ECG` si le nom le dit (IO, SO, LO, HEOG, VEOG inclus) ; sinon `EEG` si MNE le type EEG (position connue, ou unité en µV dans un fichier sans positions) ; sinon vide |
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
  sont tous optionnels. Absents ou illisibles, ils ne sont pas affichés et rien ne s'arrête. MNE, lui, s'arrête sur un tableau de filtres
  coupé ou sur un autre nom de canal (mesuré) : l'enregistrement est alors ouvert par une copie de l'en-tête sans le `[Comment]`, et
  `comment_warning` le dit. Même chemin pour un `.vhdr` qui écrit `$b`, que MNE ne connaît pas ;
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
- **Tests** : `tests/test_brainvision_tools.py` écrit ses enregistrements lui-même, sans toucher à `tests/synthetic.py` (qui relancerait
  toute la chaîne). Test d'équivalence : le même signal en EDF et en BrainVision donne les mêmes métriques dans l'outil 6, à la
  tolérance près. Puis lancement sur les vraies données en `%TEMP%`.
- **Tests proposés** : standard sur `tool1`, `tool2`, `tool3`, `tool6`.
- **Précisions apportées en faisant** :
  - le high-pass n'est coché que si le `[Comment]` dit DC : un champ vide ne dit rien, et un EDF au champ vide reste comme avant ;
  - la résolution n'est vérifiée que pour des échantillons entiers (un float32 n'a pas de pas fixe, et les AUX annoncent 1 mV) ;
  - **corrigé pour tous les formats** : dans les outils 1 et 2, le premier fichier illisible arrêtait tout le scan (`except ... as e`
    effaçait le compteur de boucle `e`) ;
  - **corrigé pour tous les formats** : l'outil 2 proposait `Tp9`, `Ft10`... au lieu de `TP9`, `FT10` (noms absents de sa liste 10-10).

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
- **Précisions apportées en faisant** :
  - **SleepFM ne peut pas tourner sans ECG ni sans ceinture** : il lui faut un canal par modalité. Les comparaisons « sans » sont
    remplacées par ECG tel quel contre ECG filtré 0,5-40 Hz (le 50 Hz retiré), et j'ai ajouté EOG IO-TP9 contre IO-Cz pour juger la
    décision 8. Sans l'alias de la ceinture, SleepFM ne pouvait pas scorer ces enregistrements ;
  - **référence en ligne recréée seulement si une dérivation en a besoin** (Cz sur les deux mastoïdes, référence moyenne) : la recréer
    à la lecture chargeait les 64 canaux en mémoire ;
  - **un EOG est « sur une entrée EEG »** quand le `.vhdr` lui donne l'unité µV et aucun gradient (les entrées AUX en ont un). Sinon,
    et pour tout EDF, il reste tel qu'enregistré ;
  - **électrodes proposées** : sans les mastoïdes (M1, M2, A1, A2, TP9, TP10) ni les canaux de contexte de l'outil 2 ;
  - **calage des seuils** (EDF de test, 2 nuits Curry, 3 enregistrements BrainVision) : ceux du plan tiennent (3 fois, 0,1 fois, 20 %,
    100 µV). TP9 du 3e enregistrement est juste sous le seuil (18 %) ;
  - **EDF inchangés** : l'outil 3bis avant et après, sur les deux études de `score_edf_equivalence` et les trois modèles, donne les mêmes
    fichiers de données. Seuls changent deux textes : la note « recording start time unknown » et le libellé « Recording start ».
- **Résultats des comparaisons** (2026-11-12, 3 siestes de 330 à 391 époques, environ 95 % d'éveil ; l'expert score 13 N1 et 37 N2
  en tout, ni N3 ni R ; kappa sur les 3 enregistrements réunis, l'exactitude ne dit rien ici car « tout en éveil » ferait 96 %) :

  | Modèle | Variante | Kappa | Par enregistrement |
  |---|---|---|---|
  | GSSC | défaut (C3, C4, F3, F4 + IO-TP9) | 0,28 | 0,49 / 0,14 / 0,65 |
  | GSSC | EOG tel qu'enregistré (IO-Cz) | 0,31 | 0,51 / 0,18 / 0,62 |
  | GSSC | EEG seul | 0,29 | 0,43 / 0,17 / 0,62 |
  | YASA | sans EMG (décidé pour l'étude) | 0,22 | 0,29 / 0,13 / 0,47 |
  | YASA | EMG tel quel (x100) | 0,21 | 0,23 / 0,15 / 0,37 |
  | YASA | EMG corrigé (/100 dans une copie du `.vhdr`) | 0,24 | 0,44 / 0,15 / 0,42 |
  | YASA | sans EMG, IO-Cz | 0,24 | 0,37 / 0,15 / 0,46 |
  | SleepFM | défaut (ECG tel quel, IO-TP9) | 0,13 | 0,10 / 0,04 / 0,39 |
  | SleepFM | ECG filtré 0,5-40 Hz | 0,12 | 0,08 / 0,05 / 0,37 |
  | SleepFM | IO-Cz | 0,14 | 0,10 / 0,05 / 0,46 |

  - **EMG de YASA** : à la mauvaise échelle, il dégrade 2 enregistrements sur 3. Corrigé, il aide sur le 1er et pas sur le 3e. Rester
    sans EMG tant que le gain n'est pas confirmé, puis corriger la résolution du `.vhdr` ;
  - **ECG de SleepFM** : le filtrer n'apporte rien, il reste tel quel (comme Score_EDF) ;
  - **référence de l'EOG** : IO-TP9 n'apporte rien de mesurable ici, un peu moins bien en tout pour les trois modèles. Sur 50 époques
    de sommeil, l'écart tient à quelques époques. TP9 est bruité sur 18 % des époques du 3e enregistrement, celui où l'écart est le
    plus grand pour SleepFM. **Décidé (2026-11-12) : la règle est gardée**, pour suivre l'apprentissage des modèles ;
  - **le 2e enregistrement** est mauvais pour les trois modèles, qui y voient beaucoup de sommeil là où l'expert score de l'éveil.
    Sans son C3 signalé, GSSC passe seulement de 0,14 à 0,15 ;
  - **SleepFM** est le plus faible sur ces siestes : il a appris sur des nuits entières et prédit trop de N2.
- **Outil 6, ta demande du 2026-11-12** (fin de phase) :
  - **(a) fait** : chaque canal comparé au canal médian de la même époque, toute la nuit, en trois colonnes à part
    (`noisy_vs_median_pct`, `flat_vs_median_pct`, `comparison_alert`). Ce n'est jamais un drapeau, donc l'outil 7 ne décoche rien de
    plus et les drapeaux EDF ne bougent pas ;
  - **(b) fait, avec ton accord** (il change des résultats EDF) : `flat_pct` ne compte que les moments où le signal reste immobile au
    moins 50 ms. Mesuré : tous les défauts injectés restent signalés, les 9 fausses alertes du 3e enregistrement disparaissent, et
    4 canaux EDF perdent leur drapeau (les deux EOG d'une nuit de test à 256 Hz, C3 et un EOG d'une autre à 512 Hz), sans plat
    réel de 50 ms. Sur la chaîne de test, seul le Fp1 écrêté change de raison (4,9 % → 1,9 %), il reste signalé par ses autres
    critères : l'outil 7 lit les mêmes canaux. Les références de l'outil 6 sont comparées hors `flat_pct` par un test dédié.

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
- **Précisions apportées en faisant** (2026-11-12) :
  - **un seul endroit pour l'heure de début** : `recording_lib.read_start_datetime` (le `.vmrk`, `None` pour un enregistrement mis en
    pause), utilisé par l'export des événements des outils 4, 5, 7, 0 ;
  - **les lectures d'`edf_io_lib`** (EEG, canaux de contexte, bornes physiques) passent par `recording_lib`. L'outil 7 appelle
    désormais `read_context_raw` au lieu de sa copie en ligne : l'EOG du compagnon reçoit la référence de l'outil 2 ;
  - **la règle de référence de l'EOG** est passée dans `recording_lib`, pour que l'outil 7 l'utilise sans charger les modèles ;
  - **Cz dans l'outil 7** : l'outil 6 la laisse de côté (plate telle qu'enregistrée). L'outil 7 la remet quand l'outil 2 déclare une
    référence. Sans référence (ta configuration actuelle), elle reste plate : c'est ce que l'outil 2 signale déjà ;
  - **bug trouvé sur tes données** : MNE type IO en `misc` (pas de position dans le fichier), et le spectre de l'outil 7 le laissait de
    côté, d'où un plantage sur les trois enregistrements. Tous les canaux du montage sont maintenant typés EEG, comme pour un EDF ;
  - **corrections de l'équipe** (FIX #8, #10 dans `edf_io_lib`, arrivées par le merge du 2026-11-12) reportées dans la copie de l'outil 7,
    sans quoi le test de synchronisation échouait.
- **Sur tes données (copie en `%TEMP%`)** : l'outil 4 détecte `_PSG4_ScoredEvents.txt` et lit les deux libellés.
  `Micro-éveil 1 ARO SPONT` est reconnu, `Micro-éveil 5 MSE` reste à choisir à la main. L'outil 5 tourne : lumières aux bornes de
  l'enregistrement (aucune autre source), TIB de 165 à 196 min, TST de 5 à 12 min. L'indice d'éveils (55 à 158 par heure) n'a pas de
  sens sur si peu de sommeil. L'outil 7 tourne en 13,5 min pour les trois enregistrements (64 canaux, compagnon EOG/EMG/ECG,
  événements placés). L'outil 8bis pré-coche tous les stades (médiane de 13 époques N2), et l'outil 8 charge et enregistre.

### Phase 4 bis : deux ajouts demandés le 2026-11-12 (avant la phase 5)

**A. La référence en ligne sans re-référence (outils 7 et 0).** Décidé : laissée de côté.

- **Constat** : une référence déclarée dans l'outil 2 est déjà retirée après le re-référencement (`preprocess_raw`), donc jamais
  signalée. Reste la référence en ligne d'un BrainVision (Cz) quand l'outil 2 ne déclare aucune référence : elle vaut 0 partout et
  elle est signalée dans toutes les époques. Sur tes données, l'outil 6 n'avait pas tourné sur la copie, et l'outil 7 a pris le
  montage de l'outil 2, Cz comprise.
- **Règle, tous formats** : quand l'outil 2 garde la référence en ligne dans le montage sans déclarer de référence, elle est laissée
  de côté : ni lue, ni dans le `.fif`, ni signalée. Une ligne le dit au scan et dans le log. Rien ne change pour un EDF (pas de
  référence en ligne) ni quand une référence est déclarée (Cz redevient un vrai signal, gardée comme aujourd'hui).
- **Une fonction** `recording_lib.flat_online_reference(path, cfg_entry)` (le nom harmonisé, ou `None`), appelée par l'outil 7 (scan)
  et l'outil 0 (chargement).
- **Outil 0, trouvé en préparant** : son étape 5 (l'outil 6) garde Cz plate et la signale, alors que l'outil 6 la laisse de côté.
  L'étape 5 la laissera de côté comme l'outil 6, et l'étape 6 la gardera seulement avec une référence déclarée, comme l'outil 7.
- **Test** : dans `test_brainvision_chain`, sans référence, Cz absente du `.fif` et la ligne présente. Avec référence, rien ne change.

**B. Déclarer les fichiers d'événements d'une étude (outils 4, 5, 7, 0).** Décidé : une déclaration pour l'étude.

- **Constat** : chaque outil cherche le TXT, le CSV et le XML, et dit ce qui manque. Sur une étude qui n'a que des TXT, l'outil 4
  avertit en orange « No event CSV detected… rely on the XML fallback », et l'outil 7 parle de `*.edf.XML` sur une étude
  BrainVision.
- **Où** : dans l'outil 4, l'outil des événements. Sous les suffixes, une ligne « Event files of this study » avec trois cases
  (*TXT export*, *CSV*, *XML*), pré-cochées d'après la détection (cochée = au moins un enregistrement en a un) et modifiables.
  Elles sont enregistrées au *Save* dans `config_param/file_names.json` (clé `event_files`, avec le suffixe choisi pour le TXT et
  le CSV). Une étude sans aucun événement : tout décoché, et *Save* enregistre seulement la déclaration, sans `event_remap.json`.
- **Lecteurs** (outils 5, 7, 0, et l'outil 4 à la réouverture) :
  - un type déclaré absent n'est plus cherché, même si un fichier existe, et son champ de suffixe et ses messages disparaissent ;
  - le suffixe enregistré est pré-choisi, avant la détection ;
  - sans aucun type, l'outil 7 masque la méthode *event* avec une ligne qui dit pourquoi, et l'outil 5 écrit « no scored events in
    this study (declared in tool 4) » au lieu de son avertissement (indices d'événements à NaN, comme aujourd'hui).
- **Sans déclaration, rien ne change** (sorties et textes identiques). Une déclaration conforme au disque donne les mêmes sorties.
- `load_events` / `event_companion_paths` : un type absent (suffixe `None`, `xml=False`) n'est pas cherché. Les copies des outils 4 et
  7 restent identiques (`test_edf_io_sync`).
- **Tests** : une étude déclarée sans CSV ni XML (aucun message sur eux dans 4, 5, 7, mêmes sorties) ; une étude déclarée sans
  événements (l'outil 7 sans méthode *event*, l'outil 5 avec la ligne d'information).
- **Décidé** : le résumé Compumedics (`_Summary_Export.txt`, source des lumières de l'outil 5) suit la même déclaration, avec une
  quatrième case.
- **Précisions apportées en faisant** (2026-11-12) :
  - **un bouton *Save the declaration*** dans l'outil 4, à côté des cases : une étude sans événements n'a rien à harmoniser, donc
    son *Save* de la section 4 restait inaccessible. La déclaration part aussi avec le *Save* de la correspondance ;
  - **le résumé a son champ de suffixe** dans l'outil 4 (détecté sur les `.txt` qui contiennent `summary`), pour que la déclaration
    le porte ;
  - **une case décochée alors que des fichiers existent** : l'outil 4 le dit en orange ;
  - **l'outil 7 retire aussi la référence plate d'un participant déjà traité** (ses canaux relus dans le fichier de paramètres) :
    la comparaison des réglages demande alors *Reprocess / Keep*.

### Phase 4 ter : corriger le gain d'un canal de contexte dans l'outil 2 (demandé le 2026-11-12, confirmé avec ses tests)

**Pourquoi** : l'EMG et l'ECG de cette étude passent par un adaptateur BIP2AUX (gain 100, confirmé sur le matériel) que le workspace
ne compense pas. Plutôt que de modifier les `.vhdr`, la correction se règle dans les notebooks, pour tout format.

**Décidé avec toi** : dans l'outil 2, sur les canaux de contexte seulement (EOG gauche / droit, EMG, ECG, les EEG restent tels
qu'enregistrés), un nombre « diviser par » par canal (1 par défaut = aucune correction), réglé par configuration de canaux et
appliqué à tous ses participants comme les canaux de contexte, avec un avertissement qui dit si une correction semble nécessaire.

- **Outil 2, à côté de chaque menu de canal de contexte** : un champ *Divide by* (1 par défaut, de 0,001 à 10 000).
- **L'avertissement** : dès que les canaux de contexte d'une configuration sont choisis, le contrôle d'amplitude déjà écrit
  (`channel_check_lib.context_amplitude_warnings`, ECG : onde R attendue entre 0,1 et 5 mV, EMG du menton : RMS jusqu'à 100 µV) tourne
  sur les trois premiers enregistrements de la configuration (lecture du seul canal concerné, rien d'écrit). Une ligne sous les menus :
  - *ECG (ECG): R waves of 59-124 mV in 3/3 recordings, about 100 times the usual 1 mV: an adapter (e.g. BIP2AUX, gain 100) may
    amplify it before the input. Set Divide by to 100?*
  - une fois le champ à 100 : *ECG divided by 100: R waves of 0.59-1.24 mV ✓* ;
  - rien d'anormal : une ligne verte. Le contrôle ne remplit jamais le champ lui-même : il propose, tu décides.
  - Trois enregistrements par configuration pour que le scan reste rapide sur une grande étude (un EDF de nuit se lit en entier pour
    un seul canal) : la ligne le dit.
- **Enregistré** dans la configuration des canaux, bloc **additif et optionnel** `context_gain` par participant (`{"emg": 100, "ecg":
  100}`), **omis quand tout vaut 1** : un JSON sans correction reste identique octet par octet.
- **Appliqué partout où ces canaux sont lus**, avant toute dérivation (le signal enregistré est divisé, puis l'EOG reçoit la référence
  de l'EEG s'il y a lieu) :
  - l'outil 3bis (l'EMG de YASA, les canaux donnés à SleepFM, l'EOG des trois modèles), et son fichier de paramètres le note ;
  - le compagnon de contexte de l'outil 7 (`edf_io_lib.read_context_raw`) et l'outil 0, qui passent par la même fonction ; le fichier
    de paramètres de l'outil 7 le note, et sa comparaison des réglages (*Skip*) le prend en compte ;
  - les contrôles de canaux des outils 3bis et 7 : ils jugent le signal corrigé, et le disent (*after the correction set in tool 2*).
- **Rien ne change** sans correction (tous les champs à 1) : mêmes JSON, mêmes sorties (preuve par `refactor_check.py`).
- **L'outil 1** garde sa ligne d'information sur un ECG / EMG en AUX, qui renvoie désormais à ce réglage de l'outil 2.
- **Pour cette étude** : une fois le facteur réglé, YASA peut reprendre l'EMG (décision prise tant que le gain était inconnu).
- **Tests proposés** : quick (la lecture divisée, le JSON sans correction inchangé, le contrôle avant / après la correction) et standard
  sur les outils 2, 3bis, 7 et 0 (une étude BrainVision avec un ECG et un EMG 100 fois trop grands : l'avertissement, le champ, le
  JSON, puis le compagnon de l'outil 7 et l'EMG de YASA à la bonne échelle).
- **Quand** : après ton commit des phases 4 bis et 5.
- **Précisions apportées en faisant** (2026-11-13) :
  - les champs *Divide by* des EOG n'ont pas de ligne d'amplitude (le contrôle porte sur l'ECG et l'EMG du menton, comme prévu) ;
    l'EOG divisé est bien contrôlé ensuite par les outils 3bis et 7 ;
  - chaque mesure est faite une fois par enregistrement et par canal : changer le champ recalcule la ligne sans relire le fichier ;
  - un canal divisé est noté là où il est lu : les `notes` du fichier de paramètres de 3bis (`EMG (EMG) divided by 100 (tool 2)`),
    une ligne du log de l'outil 7, une note de l'outil 0 ;
  - l'outil 3bis compare aussi ce facteur dans son *Skip* (seulement la part que reçoit chaque modèle : l'EOG pour GSSC, l'EOG et
    l'EMG pour YASA, les quatre pour SleepFM).

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

**Audit fait le 2026-11-12** sur un jeu synthétique en `%TEMP%\hd5` (aucune donnée réelle) : 2 siestes de 2 h à 256 Hz, 62 EEG + IO
(sans position, comme sur tes données), Cz en référence en ligne, EMG, ECG, ceinture, une électrode morte, une bruitée, une avec du
50 Hz. Chaîne 6 → 7 → 8bis → 8 → 9 sans erreur (outil 6 en 5 min, outil 7 en 4,5 min, outil 9 en 15 min), puis chaque figure regardée.

| Outil | Constat à 62 canaux | Proposition |
|---|---|---|
| 8bis | heatmap lisible (0,22 pouce par canal, bandes du haut de hauteur fixe) | rien |
| 8 | montage par époque lisible (62 rangées, contexte EOG/EMG/ECG dessous) ; tableau du détail réduit à 8 lignes | rien pour l'affichage. Voir la ligne *topographies* |
| tous (8, 0) | **aucune topographie** : elle demande une position à tous les canaux, et l'électrode de l'œil (IO) n'en a pas. Même chose sur tes données | topographie sur les canaux qui ont une position |
| 0 | le détail affiche **les 62 lignes** du tableau des métriques : l'outil 0 n'applique pas le réglage de l'outil 8 | le même réglage que l'outil 8 au-dessus de 12 canaux (8 lignes : les canaux signalés + les pires) |
| 7 | heatmap correcte, mais **l'hypnogramme du haut est écrasé** (sa hauteur suit le nombre de canaux : les libellés des stades se chevauchent) | hauteur fixe pour l'hypnogramme, comme l'outil 8bis |
| 9 | **illisible** : chaque figure met les 62 canaux sur une seule rangée de 20 pouces (32 pixels par canal, titres et axes superposés) | une grille de petits panneaux (8 colonnes, axes partagés, légende une seule fois), pour les figures par participant et de la base |
| 6, participant | rapport de **19,7 Mo** par participant : 4 figures par canal, 258 figures | le détail par canal seulement pour les canaux signalés ou en alerte de comparaison, les autres dans le tableau. Une case *All channels* (décochée au-dessus de 12 canaux) remet tout |
| 6, participant | superposition des canaux : légende de 63 noms coupée, couleurs indiscernables | en haute densité, canaux en gris fin, seuls les canaux signalés en couleur dans la légende |
| 6, participant | nuage par électrode : noms des 62 électrodes illisibles (superposés) ; panneau *At EDF bounds* vide sur BrainVision | noms verticaux en petit ; panneau masqué sans bornes physiques (comme dans la figure voisine) |
| 6, base | une figure par électrode (61 figures, 5,1 Mo) | en haute densité, une figure par métrique avec une boîte par électrode (6 figures) |
| 1, 2, 3bis | listes et cases de 64 à 71 canaux, déjà vues dans Voila par toi aux phases 2 et 3 | confirmées lisibles par toi : rien |
| 5 | pas de signal EEG | rien |

- **Un seul seuil**, `HD_CHANNEL_THRESHOLD` (12) de `qc_rejected_epochs_lib`, pour toutes les librairies d'affichage. Rien ne change à 12
  canaux ou moins.
- **Précisions apportées en faisant** (2026-11-12) :
  - **le seuil est passé dans `edf_io_lib`** (`qc_rejected_epochs_lib` le reprend) : `qc_rejected_epochs_lib` importe déjà
    `preprocessing_lib`, qui ne pouvait donc pas l'importer en retour ;
  - **la case *All channels* de l'outil 6** s'appelle *Per-channel detail for every channel*, dans la boîte *Advanced*. Elle ne change que
    les rapports, donc ce n'est pas un réglage enregistré : avec *Skip* coché, un rapport déjà fait reste tel quel ;
  - **le panneau vide *At EDF bounds*** du nuage par électrode est masqué pour toute étude dès que la métrique n'a aucune valeur (pas
    seulement au-dessus de 12 canaux), comme la figure voisine ;
  - **l'outil 0** reçoit aussi les superpositions de l'outil 6 (canaux signalés en couleur), sans la case : son étape 5 détaille déjà un
    canal choisi à la fois dans l'explorateur.
- **Test d'inertie** : `tests/refactor_check.py` sur la chaîne synthétique (EDF, 4 canaux), avant et après. Tous les fichiers doivent être
  identiques, rapports compris. Plus un test par figure modifiée, à 62 canaux, qui vérifie la mise en page (nombre de panneaux, hauteur
  fixe de l'hypnogramme, topographie présente).

### Phase 5 bis : retours de ton inspection visuelle (2026-11-13, à confirmer)

**Décidé avec toi** : la comparaison avec le canal médian devient un signalement au-dessus de 12 canaux, à seuils fixes. La PSD en
superposition de l'outil 9 vaut au-dessus de 12 canaux. Ordre : la phase 4 ter d'abord, puis celle-ci.

**Ce que la copie d'un enregistrement a montré** (copie dans `%TEMP%`, supprimée ensuite) :
- **Aucun canal signalé par l'outil 6**, alors que 10 canaux sont bruités sur 100 % des époques par rapport au canal médian (jusqu'à
  7 000 µV d'écart-type). Aucun des cinq critères ne vise un canal trop ample : sur un EDF, un canal débranché finit écrêté aux bornes
  physiques (`bounds_pct`), et un BrainVision n'a pas de bornes.
- **CP6 bimodal dans l'outil 0 et pas dans l'outil 6** : l'outil 6 a tourné rééchantillonné à 256 Hz, l'étape 5 de l'outil 0 à 500 Hz.
  Le compte de pics dépend du nombre d'échantillons (largeur des classes de l'histogramme) : 1 pic à 256 Hz, 2 à 500 Hz.
- **L'explorateur illisible en *as recorded*** : 9 canaux EEG dépassent 10 lignes dans l'époque médiane (le pire, 13 mV, fait 88 lignes à
  150 µV par ligne), l'EMG 17 mV et l'ECG 137 mV (le gain 100 de la phase 4 ter). Changer l'échelle de l'EMG ne pouvait pas suffire.

**1. Barres de progression (tous les outils).** Aujourd'hui quatre constructions : outils 6 et 7 (500 px, légende teintée), outils 5
et 9 (620 px, légende grise), 3bis et 8 à part. La légende est placée par un calcul sur la largeur du libellé, d'où le décalage.
- Une fonction `ui_lib.phase_bar(phases)` pour la barre des étapes : le libellé à part, puis la barre et sa légende dans la même colonne,
  donc alignées par construction. La légende prend le style des outils 6 et 7 (segments teintés collés sous la barre).
- Une seule largeur pour toutes les barres (620 px, libellé compris), le texte « i / N » et celui de l'étape en cours dans la même
  colonne à droite. Outils 5, 6, 7, 9 (deux barres) et 1, 1bis, 2, 3bis, 4, 8, 8bis (une barre). `test_ui_lib.py` le garde.

**2. Outil 6.**
- **Au-dessus de 12 canaux, la comparaison devient un critère de signalement** : un canal bruité (plus de 3 fois le canal médian) ou
  presque plat (moins de 0,1 fois) sur plus de 20 % des époques entre dans `suspect_reason` et `exclude`. Il est donc décoché par défaut
  dans l'outil 7, coloré dans les superpositions, en rouge dans le nuage par électrode et dans le rapport de la base. Les seuils sont ceux
  du contrôle de l'outil 3bis (`channel_check_lib`), écrits dans la boîte des seuils, sans champ. À 12 canaux ou moins, rien ne change
  (avertissement seul). L'étape 5 de l'outil 0 suit (même fonction).
- *Skip* : le nouveau critère compte dans `flag_criteria`, donc un participant déjà traité dont les signalements changent pose la question
  *Reprocess / Keep*. Un participant traité avant le 2026-11-12 (colonnes de comparaison vides) n'a rien à comparer.
- **Une ligne visible sur la page après le scan** pour un montage de plus de 12 canaux : le rapport de chaque participant ne détaille
  que les canaux signalés, et la case *Per-channel detail for every channel* (boîte *Advanced*) les détaille tous.
- La règle de CLAUDE.md *Channel checks warn, they never decide* reçoit son exception (outil 6, au-dessus de 12 canaux).

**3. Heatmap de l'outil 7 (et de l'étape 6 de l'outil 0) : l'hypnogramme dépasse.** La barre de couleur prend sa largeur à la heatmap
seule. Elle passe dans une colonne à part, comme dans la heatmap de l'outil 8 : l'hypnogramme et les époques s'alignent. **Pour tous
les montages** (le défaut existe aussi en EDF) : la figure du rapport EDF change, rien d'autre.

**4. Figures de l'outil 8bis (décision) et de l'outil 7 (heatmap), au-dessus de 12 canaux** : lignes de 0,14 pouce au lieu de 0,22 (8bis)
et 0,2 (7), largeur plafonnée à 14 pouces, polices inchangées.

**5. Panneau de détail des outils 8 et 0 (avec ton point 8, les topographies).**
- Outil 8 : le nombre de canaux avant leur liste au chargement (`channels (62) = [...]`).
- **Le tableau des métriques sort de la figure** : un tableau HTML à côté, qui défile quand il est plus haut que la figure. Tous les
  canaux, dans l'ordre du fichier, les valeurs hors seuil en rouge. Le réglage *Table rows* disparaît (outil 8) et l'outil 0 n'a plus
  de limite de 8 lignes. Pour tous les montages : à 12 canaux ou moins, le tableau tient sans défiler. Ce panneau n'écrit aucun fichier.
- **Quatre petites topographies en 2 x 2** dans la place libérée : p-p, gradient, MAE et R² du 1/f, chacune avec ses électrodes hors
  seuil cerclées de rouge. Comme aujourd'hui, seulement au-dessus de 12 canaux qui ont une position.

**6. Outil 9.**
- Rapport individuel, *Clean epochs per stage* : le pourcentage à côté du nombre, `123 (85 %)` = époques propres / époques de ce stade
  dans l'enregistrement. Pour tous les montages (la figure du rapport EDF change).
- Rapport de la base, *Clean epochs per participant* : le pourcentage d'époques propres au-dessus de chaque barre, et un tableau par
  participant (époques de l'enregistrement, propres, %, puis propres et % par stade, et la colonne *excluded*). Pour tous les montages.
- **Au-dessus de 12 canaux**, les deux figures PSD (*Mean PSD per stage* et *after removing the aperiodic component*), dans les deux
  rapports : une superposition (un panneau par stade, chaque canal en trait fin gris, la moyenne des canaux en trait épais à la couleur
  du stade), puis une section repliée *Per-electrode panels* qui contient la grille actuelle. MNE ne sait pas replier une figure : la
  section est un `<details>` HTML avec l'image dedans.
- Le log-log : les graduations 30 et 45 Hz se chevauchent sur les petits panneaux. 30 est retiré au-dessus de 12 canaux.

**7. Outil 0.**
- **Événements** : un menu comme pour l'hypnogramme. Les fichiers trouvés à côté de l'enregistrement (TXT, CSV, XML, celui que liraient
  les outils 5 et 7 présélectionné), *none*, ou *Other file…* (un sélecteur, avec le même avertissement orange : les outils en lot ne
  le trouveront pas).
- **Étape 5** : la préparation du signal reprend celle que l'outil 6 a enregistrée pour ce participant (rééchantillonnage, passe-haut),
  avec une ligne qui le dit. Sans table de l'outil 6, la règle de l'outil 6 (passe-haut à 0,1 Hz pour un enregistrement DC). Les deux
  outils comparent alors la même chose. La sensibilité du compte de pics au rééchantillonnage est notée dans le SPEC.
- **Étape 6** : la disposition de l'outil 7, une ligne par réglage, la case à cocher puis le champ puis l'unité (« Resample to [256] Hz »
  au lieu de « Resample to Hz [256] »), les seuils alignés.
- **Explorateur** : une option *ticked in step 6* dans le menu *Channels*, valable dans tous les modes d'affichage. Les canaux décochés
  avant de lancer l'étape 6 sortent du montage. La règle « jamais écrêter » reste. L'EMG et l'ECG reviennent à la bonne échelle avec
  la phase 4 ter.

**Tests proposés** : quick (la barre des étapes et les pages dans `test_ui_lib`, le critère de l'outil 6 au-dessus de 12 canaux et son
absence à 12, l'alignement de la heatmap, les tailles 8bis / 7, la superposition et la section repliée de l'outil 9, les pourcentages,
le tableau HTML et les 4 topographies, le menu des événements de l'outil 0), puis standard sur les outils 0, 6, 7, 8, 8bis, 9.
`refactor_check.py` sur la chaîne EDF : tout identique sauf les changements voulus, listés dans `INTENDED_CHANGES` (figure de la heatmap
du rapport de l'outil 7, figure et tableau des époques propres des rapports de l'outil 9).

**Précisions apportées en faisant** (2026-11-14) :
- les rapports HTML ne sont pas comparés au golden (seulement les tables) : `INTENDED_CHANGES` n'a rien à recevoir.
  `refactor_check.py` montre les 9 rapports changés de la chaîne synthétique (4 de l'outil 7, 5 de l'outil 9), toutes les tables
  identiques ;
- les jumeaux Curry, gelés, passent encore une largeur de barre et un nombre de lignes de tableau : ces deux paramètres sont gardés,
  ignorés, jusqu'à la phase 6 ;
- outil 0, menu des événements : le choix vaut pour ce que montre l'explorateur ; les étapes 4 et 6 lisent les événements comme les
  outils 5 et 7, pour que leurs résultats restent les leurs (dit sous le menu) ;
- outil 0, étape 5 : la préparation de l'outil 6 est reprise au choix du participant, et une ligne la dit ;
- la superposition de l'outil 9 trace les canaux en gris clair : le gris de l'éveil, celui de la moyenne, se confondait avec eux.

### Phase 5 ter : deuxième inspection visuelle (2026-11-17, faite)

**Décidé avec toi** : le bouton Stop passe dans une phase à part (6 bis, avec `jupyter-ui-poll`), la numérotation des dossiers de
rapports après la phase 6 (6 ter), l'interpolation des canaux rejetés plus tard, hors de ce plan (noté dans les modules prévus du SPEC).
Pour la lenteur de l'outil 0, on mesure d'abord.

**Réponses sans changement de code** :
- **Outil 1, premier fichier plus lent** : normal. Le premier fichier charge les parties de MNE, SciPy et matplotlib qu'il utilise.
- **Outil 8, boîtes rouges de la section 2** : la distribution des époques rejetées (bleu = gardées). Le titre de la figure le dit.
- **SleepFM sur les 64 électrodes** : non. Il ne reconnaît que les noms de sa liste de canaux (`channel_groups.json`) et met en commun
  les canaux d'une même modalité. Il a été entraîné sur des PSG cliniques de quelques EEG : 64 électrodes ne lui apporteraient pas
  d'information spatiale et sortiraient de son entraînement. Il garde les dérivations AASM présentes (F3, F4, C3, C4, O1, O2).
- **Outil 9, minimum de 20 époques propres** : il vaut **par participant**, pour chaque case participant × stade × canal. Une case
  sous le minimum vaut NaN et le participant manque à la figure de groupe de ce stade. Le champ *Min epochs per cell* existe déjà :
  le baisser puis *Re-aggregate only* inclut les siestes sans recalculer les PSD.
- **Interpolation** : aucun outil n'interpole. Les outils 8 et 8bis écartent les canaux, l'outil 9 les calcule mais les laisse hors des
  figures et des statistiques de groupe.

**1. Bug : un stade absent et un canal décoché (outils 7 et 0).** `preprocessing_lib.set_bads_aside` recrée les époques des canaux
décochés avec les codes de tous les stades (W à R). MNE refuse un code sans époque (`No matching events found for N3`). Tout participant
sans un des stades et avec un canal décoché échoue, dans l'outil 7 comme à l'étape 6 de l'outil 0. Correction : `on_missing='ignore'`
dans cet appel. Les autres cas ne changent pas. Test : un enregistrement synthétique sans N3, un canal décoché, outils 7 et 0.

**2. Barres de progression (tous les outils).**
- **Texte tronqué** quand la page est étroite : le texte de droite passe sous la barre et revient à la ligne au lieu d'être coupé. ==> a validé après inspection du résultat, j'ai peur que le rendu ne soit pas terrible.
- **Nouvelle règle (SPEC, rappel court dans CLAUDE.md)** : une barre ne reste jamais immobile pendant qu'une étape travaille.
  - Une étape qui boucle (canaux, époques, morceaux, figures) fait avancer la barre à chaque élément.
  - Une étape qui ne peut pas le dire (une lecture MNE, la prédiction de YASA) affiche un petit signe animé à côté de son nom. Le
    navigateur l'anime lui-même, donc il bouge même quand Python est occupé.
- **Où** :
  - 3bis : une barre d'étapes par participant, un segment par modèle. Les combinaisons de GSSC et les morceaux de SleepFM la font
    avancer. La barre des participants compte les participants.
  - 9 : la PSD par blocs d'époques, les bandes, puis le rapport figure par figure.
  - 0 : le chargement de l'étape 1 et chaque étape.
  - 5, 6, 7, 8, 8bis : vérifiés sur l'enregistrement synthétique de 62 canaux (point 8), corrigés là où la barre reste immobile.

**3. Journal des runs coloré.** Les `⚠` que tu as vus sont des lignes du journal (`print`), en texte brut. Une fonction
`UI.log(kind, text)` les écrit en couleur, avec les codes ANSI que Voila affiche en couleur : `⚠` orange, `❌` rouge, `✅` vert, comme
les messages de la page. Toutes les pages l'utilisent. Aucun outil n'écrit son journal dans un fichier, donc aucune sortie ne change.

**4. Outil 3bis.**
- **Menus limités à ce que l'outil 2 déclare.** Aujourd'hui les modes EOG sont fixes. Avec un seul EOG déclaré (à droite), GSSC propose
  encore *L − R bipolar* par défaut, et YASA *Left*.
  - EOG : seuls les modes que les canaux de contexte permettent (un EOG droit seul : GSSC *Right only* / *None*, YASA *Right* /
    *None*), avec le côté présent par défaut.
  - Avec plusieurs configurations, une option absente de certaines est marquée `(n/N)`. Cela vaut aussi pour les électrodes EEG et
    l'EMG.
  - Une ligne sous les menus nomme les enregistrements où un canal choisi manque, et dit ce que le modèle fait alors (règles déjà en
    place).
- **Rappel avant le run** : une ligne SleepFM, ses EEG (dérivations présentes), EOG, EMG, ECG et respiration trouvés, par configuration.
- **Progression** : point 2 (la barre n'avançait qu'à la fin du premier modèle).

**5. Outil 8bis, au-dessus de 12 canaux** : la figure de décision tient sur une page. L'hypnogramme passe de 1,0 à 0,6 pouce et la
bande des époques de 0,4 à 0,2 pouce.

**6. Panneau de détail des outils 8 et 0.**
- **Topographies.** Aujourd'hui, l'échelle de couleur va du minimum au maximum de l'époque. Un seul canal très mauvais écrase les
  autres dans une seule couleur, et sur le crête-à-crête un canal plat est cerclé sur une valeur proche de zéro.
  - L'échelle est centrée sur le seuil (le seuil au milieu, saturée au-delà), et le seuil est marqué sur la barre de couleur.
  - Les cercles sont rouges au-dessus du seuil et cyan sous le seuil de plat (carte p-p). Le titre de chaque carte dit sa règle.
- **PSD face à la référence propre.**
  - La référence est tracée au-dessus des canaux, dans un style qu'on ne confond pas avec l'époque : trait noir et son IQR en
    deux traits noirs pointillés (plus de remplissage gris dessous).
  - L'époque garde sa bande gris clair et sa médiane grise.
  - Pour tous les montages : à 12 canaux ou moins, les traits gris des canaux restent et la référence passe dessus.

**7. Outil 9.**
- **Au-dessus de 12 canaux, chaque figure dessinée canal par canal** devient une figure agrégée sur les canaux gardés, suivie d'une
  section repliée avec les panneaux par électrode (comme la PSD en phase 5 bis).
  - Rapport individuel : log-log avec l'ajustement, paramètres apériodiques au fil de la nuit, puissance par bande et par stade.
  - Rapport de la base : puissance par bande entre participants, paramètres apériodiques par stade.
- **Titre tronqué** (*after removing the aperiodic component*) : les titres des figures reviennent à la ligne à la largeur de la figure.
- **Minimum d'époques par case.**
  - Au scan, une ligne par stade dit combien de participants atteignent le minimum (et la médiane de leurs époques propres). Sous
    le minimum, la ligne propose de baisser *Min epochs per cell*.
  - Le rapport de la base nomme les stades absents de ses figures et dit pourquoi.
  - La valeur ne change pas seule : c'est un choix de méthode.

**8. Outil 0.**
- **Mesures d'abord** : un enregistrement synthétique de 62 canaux à 500 Hz (la fréquence de l'étude), durée à confirmer avec toi.
  - Je chronomètre l'étape 1 (lecture, PSD de chaque époque × canal, spectrogramme du navigateur) et chaque étape face à son outil
    séparé.
  - Les résultats vont dans ce plan, et la correction t'est proposée avant d'être codée.
- **Explorateur** :
  - Un menu *Hypnogram shown* : celui en usage (modifiable, par défaut) ou un autre trouvé à côté de l'enregistrement (scorings de
    3bis, celui de l'étape 3). Un autre est en lecture seule : le rescoring est coupé, une ligne le dit.
  - Tant que l'étape 6 n'a pas tourné, une ligne dit que le montage se lit mieux après elle (*Signal: preprocessed*, *Channels:
    ticked in step 6*).
- **Étape 5** : la légende de chaque canal coloré dit pourquoi il est signalé (`C3: flat_vs_median 85 %`). Fonction de `quality_lib`,
  donc les superpositions du rapport de l'outil 6 la gagnent aussi, pour tous les montages.
- **Étape 6** : *Tick all* et *Untick all* à côté de *Untick step 5's flagged channels*. L'erreur `N3` est le point 1.

**Tests proposés** : quick (le bug du point 1, `UI.log`, le retour à la ligne des barres, les menus de 3bis avec un seul EOG, les
échelles et cercles des topographies, les figures agrégées et repliées de l'outil 9 et leur absence à 12 canaux, la ligne du minimum
d'époques, le menu d'hypnogramme et les boutons de l'outil 0), puis standard sur les outils 0, 3bis, 6, 7, 8, 8bis, 9. `refactor_check.py`
sur la chaîne EDF : toutes les tables identiques. Changent les rapports des outils 6 (légende des canaux signalés) et 9 (titres).
Le panneau de détail des outils 8 et 0 est seulement à l'écran. Puis ton inspection visuelle.

**Précisions apportées en faisant** (2026-11-17) :
- le signe animé est mis par `ui_lib` sur tout texte d'étape d'une barre d'étapes : les outils 5, 6, 7, 8 (barre du rapport) et 9
  l'ont donc sans autre changement, l'outil 3bis sur sa nouvelle barre *Models*, l'outil 0 sur les lignes d'état de ses étapes ;
- outil 9 : le lissage avant l'ajustement 1/f n'avait aucune progression, et l'ajustement en parallèle (au-dessus de 5 000
  ajustements, donc toujours sur un casque) non plus. Les deux avancent maintenant (par blocs de 25 époques, et par paquets du
  calcul parallèle, quatre par cœur), avec les mêmes nombres ;
- outil 3bis : une étude sans EOG déclaré enregistre maintenant `eog_mode` *None* (mêmes signaux qu'avant : le mode bipolaire par
  défaut ne trouvait aucun EOG) ;
- outil 0 : le menu EOG de GSSC à l'étape 3 garde ses modes fixes (un seul participant, ses canaux de contexte affichés à
  l'étape 1) ;
- les menus de l'outil 3bis et les étiquettes `(n/N)` gardent comme valeur le nom seul : les tests et l'enregistrement des
  réglages ne changent pas.

**Mesures de l'outil 0** (2026-11-18, enregistrement synthétique : 62 EEG + EOG + EMG + ECG, 500 Hz, 90 min, dans `%TEMP%`) :

| Étape de l'outil 0 | Durée | Outil séparé | Durée |
|---|---|---|---|
| 1 (chargement) | 18 s | — | — |
| 5 (qualité) | 154 s | 6 | 165 s |
| 6 (prétraitement) | 53 s | 7 | 66 s |
| 7 (décision) | 3 s | 8bis | 6 s |
| 8 (spectral) | 40 s | 9 | non comparable* |

\* Sur un bruit pur, l'outil 8bis écarte tous les canaux et exclut le participant, que la sélection de l'outil 9 laisse de côté.

- **Les étapes ne sont pas plus lentes que les outils séparés** (un peu plus rapides : elles n'écrivent pas de rapport). L'impression
  vient de l'étape 5, longue en elle-même sur 62 canaux (les mêmes calculs que l'outil 6).
- **L'étape 1 (18 s)** : lecture 3 s, signal de chaque époque en µV 1 s, **PSD de chaque époque × canal 8 s** (pour le panneau de
  détail), le reste (canaux de contexte, spectrogramme, premier dessin) environ 6 s. **La mémoire** : le signal est tenu deux fois en
  float64 (l'enregistrement pour les étapes 5 et 6, sa copie par époque pour l'explorateur), 1,4 Go chacun ici, 2,5 Go pour le
  kernel. Sur une nuit de 8 h, ce serait environ 95 s et 14 Go : la mémoire d'un PC de 16 Go déborde sur le disque, et tout ralentit.

**Correction retenue (A + B, confirmée le 2026-11-18)**, affichage seulement (les étapes 2 à 8 lisent l'enregistrement comme avant) :
- **A, la PSD du panneau de détail à la demande** : l'étape 1 n'en calcule plus. La PSD d'une époque se calcule quand elle s'affiche,
  la référence propre de son stade la première fois que ce stade s'affiche, puis elles restent jusqu'au prochain changement de
  signal. La référence prend au plus 3000 spectres (époques × canaux), ses époques réparties sur la nuit : toutes celles d'un stade
  d'un PSG de 6 canaux jusqu'à 500, 48 sur un casque de 62 canaux. Sa légende dit `n=48 of 412 epochs`. Chaque spectre ne dépend
  que de son époque : les valeurs sont celles d'une PSD de toute la nuit (vérifié, identiques en brut, à 1e-15 près avec le lissage).
- **B, le signal de l'explorateur en float32**, lu 50 époques à la fois (aucune copie float64 de la nuit). L'outil 8 reste en
  float64. L'affichage prétraité ne recopie plus les époques de l'étape 6 : elle a déjà mis ses canaux décochés de côté.

Mesuré après (même enregistrement de 90 min, PC en usage pendant la mesure : les durées sont relatives) :

| | Avant | Après |
|---|---|---|
| PSD de chaque époque × canal (étape 1) | 8 à 10 s | aucune |
| Référence d'un stade, à sa première apparition | — | environ 2 s (6 s avec le lissage de l'outil 7) |
| Redessin d'une époque, référence déjà calculée | environ 2,6 s | inchangé (62 ajustements 1/f, le montage) |
| Mémoire après la copie d'affichage | 2,9 Go | 2,2 Go |
| Pic de mémoire du chargement | 6,0 Go | 3,0 Go |

Sur une nuit de 8 h, la référence reste à 2 s par stade (48 époques quelle que soit la durée), au lieu d'environ 50 s de PSD au
chargement, et le pic de mémoire passe d'environ 32 Go à environ 16 Go (la lecture de l'enregistrement en float64 reste le plus gros
poste).

### Phase 5 quater : troisième inspection visuelle (2026-11-18, confirmée, codée)

Des maquettes sur données synthétiques (62 et 4 canaux) montrent chaque proposition à côté de la figure actuelle. Elles sont dans le
scratchpad de la session, pas dans le dépôt.

**1. Barres de progression (`ui_lib`, toutes les pages).**
- **Noms des étapes coupés** (*Figur*, *HTM*) : chaque segment de la légende devient au moins aussi large que son nom. Les autres
  segments se partagent le reste selon leur coût. La barre bleue suit les mêmes largeurs, donc elle reste alignée sur la légende. Les
  outils comptent toujours dans leurs propres unités : la conversion est faite dans `phase_bar`, aucun outil ne change pour cela.
  Maquette : `6_phase_bar.html` (ou sa capture `6_phase_bar.png`).
- Le texte de droite qui passe sous la barre sur une page étroite (phase 5 ter) reste : il ne gêne pas.
- **Signe qui tourne après *Done.*** (outil 8, section 2, barre du rapport). L'outil écrit son dernier texte après `finish_progress`,
  qui avait déjà arrêté les signes, donc le texte final en reçoit un nouveau. Correction : le texte final d'abord (*Done.*, *HTML not
  saved.*, *Failed.*), puis `finish_progress`. Les autres pages écrivent dans le bon ordre (vérifié).
- **Textes à droite des deux barres alignés.**
  - Aujourd'hui, le texte de la barre des participants est centré sur la hauteur de la barre, en taille normale. Celui de la barre
    d'étapes est collé en haut, en petit. Dans l'outil 3bis, *1 / 3* et *sub01: GSSC* ne sont donc pas à la même hauteur.
  - Les deux textes prennent le même bloc : même taille, première ligne centrée sur sa barre.
  - Les outils 6 et 7 ajoutent après la barre une étiquette *Participant X (i/N)*, décalée elle aussi. Elle disparaît : la barre dit
    déjà *i / N*, et le texte d'étape nomme le participant (*sub01 · loading…*). Le *Done. (durée)* de l'outil 7 passe dans le texte
    d'étape.
  - Je vérifie le résultat sur une capture de la page Voila (Edge sans fenêtre, sur la page locale : rien n'est envoyé).

**2. La PSD de l'époque affichée (outils 8 et 0) : la médiane des époques propres manque dans l'outil 8.**
- **Pourquoi** : une époque compte comme propre seulement si aucun de ses canaux n'est signalé (la règle *any* de l'outil 7). Sur 62
  canaux, presque aucune époque de veille ne passe. Sous 3 époques propres, la figure ne trace pas la médiane, sans le dire. Ta
  capture de l'outil 0 était sur le signal brut, où rien n'est signalé : les 320 époques W y comptaient comme propres. Sur le signal
  prétraité, l'outil 0 aurait le même manque.
- **Correction, au-dessus de 12 canaux** : la référence est faite des canaux propres de chaque époque du stade (les paires époque ×
  canal non signalées, les canaux écartés exclus). Chaque époque dont au moins la moitié des canaux sont propres donne la moyenne de
  ses canaux propres, puis la médiane et l'IQR se prennent sur ces époques.
  - Outil 8 : avec les méthodes cochées en section 1, comme sa carte des paires.
  - Outil 0 : sur le signal prétraité, avec les signalements de l'étape 6. Sur le signal brut ou filtré, rien n'est signalé : toutes
    les époques du stade, comme aujourd'hui.
  - Légende : *clean W median (clean channels of 47 of 320 epochs)*.
  - À 12 canaux ou moins, rien ne change (les époques sans canal signalé).
- **Si la référence ne peut pas être faite** (moins de 3 époques), une ligne dans la figure le dit, avec le nombre d'époques propres.
- La médiane reste en noir, comme dans l'outil 0. **La légende passe sous le graphique**, sur deux colonnes : elle ne déborde plus.
- La même règle vaut pour les PSD par stade de la section 2 de l'outil 8 (et de son rapport) : leur médiane des époques propres manque
  pour la même raison sur un casque.

**3. Topographies des outils 8 et 0.** Le rendu étrange vient de l'interpolation. MNE peint l'espace entre les électrodes, puis
jusqu'au bord de la tête. Quand deux électrodes voisines ont des valeurs très différentes, la surface entre elles dépasse les deux
valeurs : ce sont les intersections que tu vois. La transition est plus marquée pour l'amplitude, car une électrode au-delà de deux
fois le seuil sature la couleur. Sur la maquette, quatre électrodes temporales sous le seuil de R² colorent tout le bord gauche et
droit de la carte (ligne A). Trois propositions, sur deux époques synthétiques : `%TEMP%\dtk_q5\topomaps_epoch1.png` et
`topomaps_epoch2.png` (à décider) :
- **B** : les mêmes cartes interpolées, avec les couleurs du seuil (pâle sous le seuil, orange puis rouge au-dessus).
- **C (je recommande)** : chaque électrode est un disque de sa couleur, sans interpolation, avec les couleurs de B.
  - La valeur montrée est celle de l'électrode : rien n'est inventé entre elles.
  - Les cercles rouges (au-dessus du seuil) et cyan (plat) ne changent pas.
- **D** : B, avec une extrapolation limitée à la zone des électrodes.

**4. Hypnogrammes : une seule ligne partout.**
- **Outil 7** : le titre passe sur l'axe de l'hypnogramme, comme dans l'outil 8. Sur un casque, il ne flotte plus à un pouce
  au-dessus. Pour tous les montages.
- **Outil 8bis** : le style de l'outil 7, c'est-à-dire plus de cadre, un trait gris `#555555` de 1,2, REM en rouge à la même
  épaisseur, et plus d'étiquette *stage*.
- Maquettes : `3_tool7_heatmap_*_proposed.png`, `4_tool8bis_*_now.png` et `4_tool8bis_*_proposed.png`.
- **Une fonction `edf_io_lib.draw_hypnogram`**, à côté de `custom_stage_style`, trace la ligne : couleur, épaisseur, REM et stades
  personnalisés, axe de gauche seul avec les noms des stades. Chaque figure garde ses axes (époques ou heures) et ses autres panneaux.
  - **Tous les hypnogrammes (décidé)** : les bandeaux au-dessus d'une matrice d'époques (outil 7, outil 8 pour sa carte et sa bande de
    revue, outil 8bis), 3bis et sa comparaison, 5, 6 et 0 au-dessus du spectrogramme, 9 (époques propres). Aujourd'hui ils varient :
    noir de 1,0 (6, 0), gris de 1,1 avec cadre complet (5), 1,2 avec grille (3bis). Le SPEC demande déjà un seul style partout
    (*Display conventions*, 2026-10-13).

**5. Outil 9 : paramètres apériodiques.**
- **Au fil de la nuit** (rapport individuel) : des points pleins, plus gros (alpha 0,9 au lieu de 0,6, trois fois la surface).
  Maquettes : `5_tool9_night_now.png`, `5_tool9_night_proposed.png`.
- **Par stade** (rapport de la base) : un point plein par participant et la médiane en trait noir, au lieu des boîtes pâles.
  Maquettes : `5_tool9_group_now.png`, `5_tool9_group_proposed.png`.
  - **La puissance par bande de la base aussi (décidé)** : c'est le même type de figure.
- Les couleurs des stades ne changent pas : ce sont celles de toute la boîte à outils (N1 et N2 y sont deux violets proches). Je ne les
  change que si tu le demandes.

**Tests proposés**
- Quick :
  - les largeurs de la légende et l'alignement de la barre sur elle (`test_ui_lib`) ;
  - plus de signe après *Done.* dans l'outil 8 ;
  - la PSD de l'époque affichée : sur 62 canaux dont chaque époque a un canal signalé, la médiane des canaux propres est tracée ;
    à 4 canaux, la même référence qu'aujourd'hui ; sous 3 époques, la ligne qui le dit ; la légende sous le graphique ;
  - les topographies retenues ;
  - la même ligne d'hypnogramme dans chaque outil ;
  - les points pleins de l'outil 9 (paramètres apériodiques, puissance par bande de la base).
- Standard : les outils 0, 3bis, 5, 6, 7, 8, 8bis, 9.
  `test_settings_skip.py` lit l'étiquette *Done.* de l'outil 6 : il est adapté.
- `refactor_check.py` : toutes les tables identiques. Seuls les rapports des outils touchés changent.
- Puis ton inspection visuelle.

**Confirmé le 2026-11-18** : points 1, 2, 4, 5 tels quels, **topographies inchangées** (ton choix après les figures A à D).

**Précisions apportées en faisant** (2026-11-18) :
- les textes des deux barres : 12 px, la première ligne centrée sur la barre. Vérifié sur une capture de la page Voila (Edge sans
  fenêtre, page locale, serveur arrêté et dossier supprimé ensuite) ;
- l'outil 8 écrit *Done ✅* (plus *Done.*), les outils 6 et 7 finissent sur *Done ✅* / *Done in … ✅* dans le texte d'étape ;
- un hypnogramme sans stade pour une époque (époque retirée, étiquette inconnue) laisse un trou. Les outils 7 et 8 la mettaient
  sous N3, sans nom ;
- l'outil 3bis écrit *REM* au lieu de *R* sur ses hypnogrammes et n'a plus de grille, comme les autres ;
- deux superpositions gardent leur couleur de contraste : la ligne blanche du navigateur de l'outil 0 sur le spectrogramme, la
  ligne noire de l'expert sur l'hypnodensité de l'outil 3bis ;
- l'hypnogramme des époques propres de l'outil 9 devient la ligne de la boîte à outils avec des trous (plus de points colorés) ;
- sous 3 époques de référence, la ligne qui le dit vaut aussi pour les PSD par stade de la section 2 de l'outil 8, à tout
  montage.

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

**Plan détaillé (2026-11-18, à confirmer)**

**1. Le lecteur Curry, sur le modèle de BrainVision.**
- `curry_header.py` et `curry_io.py` deviennent `tools/curry_lib.py`. Le `.dpo` reste lu par le parseur maison : MNE lit tout le
  signal pour donner un en-tête (environ 15 s pour 7 Go, mesuré quand le jumeau a été écrit).
- `curry_lib.read_cdt_header(path)` rend le même dictionnaire que `read_vhdr_header` :
  - `label` : les canaux des deux groupes du `.dpo`, dans l'ordre du fichier ;
  - `transducer_type` : `EEG` pour le groupe EEG du `.dpo` (ce que faisaient les jumeaux), sinon le type déduit du nom (EOG, EMG,
    ECG, la même règle que BrainVision), sinon vide ;
  - pas de bornes physiques, donc pas de `bounds_pct` et le plafond de ±500 µV à l'affichage (les règles de la phase 2) ;
  - les impédances en kΩ (le fichier les donne en Ohm), la dernière mesure de chaque canal comme dans le jumeau, *not measured* pour
    `-1`, *disabled* pour `-2`, au format de BrainVision : le contrôle des impédances de l'outil 1 marche sans changement ;
  - les filtres : le parseur actuel n'en lit aucun (question 2).
- `recording_lib` reconnaît `.cdt` : `read_header`, `read_start_datetime` (le `.dpo`), `read_raw` (`mne.io.read_raw_curry`, puis les
  canaux demandés), `online_reference` → aucun, comme un EDF (le `.dpo` ne dit pas contre quoi les canaux sont enregistrés).
- `study_files_lib.RECORDING_SUFFIXES` gagne `.cdt`. Un `.cdt` du même nom qu'un `.edf` ou un `.vhdr` : le même avertissement.
- `curryreader` reste dans `environment.yml` : MNE en a besoin pour lire le signal.

**2. Les outils.**
- **Outil 1** : le contrôle des impédances de la phase 2 vaut pour Curry. Les sorties vont dans `reports_inspection/`, sous les noms
  des autres formats (plus de `reports_inspection_curry/` ni de suffixe `_curry`). Rien n'est renommé dans une étude existante : seules
  les données de test ont ces anciens noms.
- **Outil 2** : le masque EEG vient de `transducer_type`.
- **Outils 0, 3, 3bis, 4, 5** : rien de propre à Curry. L'heure de début qui convertit l'export texte des événements vient du `.dpo`.
- **Outil 6** : pas de bornes, donc les règles de la phase 2. Le high-pass par défaut : question 2.
- **Outil 7** : lecture et heure de début par le lecteur.
- **Outils 8, 8bis** : ils ne dépendent déjà pas du format.
- **Les copies de lecture** que les outils 2, 4, 6 et 7 gardaient pour les générateurs deviennent des imports d'`edf_io_lib`, et
  `quality_lib` importe ses fonctions de stades. Les sorties ne changent pas (prouvé par `refactor_check` sur la chaîne synthétique).
  `test_edf_io_sync.py` ne garde que la copie de l'outil 1bis. Deux TODO du SPEC se ferment : *A Curry twin of tool 0* et *EDF
  reading of tools 2, 4, 6, 7 through `edf_io_lib`*.

**3. Ce qui disparaît.**
- Dans `tools_curry/` : les 8 notebooks, les 7 générateurs, `curry_header.py` et `curry_io.py` (déplacés), `test_mne_curry_header.py`
  (une mesure ponctuelle sur données réelles, sa conclusion est déjà dans le SPEC). `tools_curry/inspect_curry.py` n'est pas versionné :
  question 3.
- `tests/test_curry_twins.py`. `test_notebooks_execute.py` et `test_ui_lib.py` ne regardent plus `tools_curry/`.
- **Documentation** : la section *Curry 9 support* du SPEC est réécrite (Curry devient un format du lecteur, comme BrainVision, avec ses
  particularités). Dans CLAUDE.md, le bloc *Curry twins* et les « re-run the generator » disparaissent.

**4. Tests.**
- **Un enregistrement Curry synthétique**, écrit par les tests (`tests/curry_files.py`, comme `brainvision_files.py`) : un `.cdt` en
  float32 et un `.dpo` minimal. Je vérifie d'abord que MNE l'ouvre. Sinon, Curry n'est testé que sur les vraies données (point 5).
- **Quick** : `test_curry_lib.py` (en-tête, impédances en kΩ, heure de début, signal relu égal au signal écrit, `recording_lib` sur un
  `.cdt`).
- **Standard** : les outils 1, 2, 6, 7 et 0 sur une petite étude Curry synthétique, comme `test_brainvision_chain.py`. `refactor_check`
  sur la chaîne EDF synthétique : tout identique.
- Puis **full**.

**5. Validation sur les deux enregistrements Curry réels** (`tools/test_data_curry`, 2 × 7 Go). C'est la seule phase où l'on peut perdre
quelque chose.
- **Avant toute modification** : une copie dans `%TEMP%\dtk_curry` (environ 14 Go, plus 25 Go de sorties, 174 Go libres). Les jumeaux
  actuels tournent dessus (outils 1, 2, 6, 7, 8bis, avec la configuration de l'outil 2 trouvée dans la copie, ou refaite), et leurs
  sorties sont gardées.
- **Après** : les outils uniques sur une copie neuve, avec les mêmes réglages, puis `refactor_check` sur toutes les sorties.
- **Différences attendues** : les jumeaux sont gelés depuis la phase 2, donc tout ce qui est venu ensuite apparaît (outil 1 : autres noms
  et colonnes ; outil 6 : colonnes de comparaison au canal médian, signalement au-dessus de 12 canaux, raisons dans les légendes ; les
  figures des rapports). Les tables des outils 7 et 8bis devraient être identiques. Toute autre différence est expliquée avant de
  continuer.
- **Ce que je vois** : des noms de fichiers et de canaux, et les différences entre tables (des valeurs de mesures). Jamais le signal.
- Run long (quelques heures), avec keep-awake. Puis ta vérification dans Voila sur Curry, comme le SPEC le demandait déjà.

**Ordre** : la sauvegarde des sorties des jumeaux d'abord (point 5), puis le lecteur et ses tests synthétiques, les outils et les
imports, la comparaison sur Curry, les suppressions et la documentation, les tests full, ta vérification.

**Questions**
1. **La validation sur les données Curry réelles** : je la lance (chaque commande sur ces dossiers te demande ton accord), ou je te
   prépare un script que tu lances, et tu me rapportes le résumé ?
2. **Le high-pass de l'outil 6 sur Curry** : le jumeau le coche (amplificateur souvent DC), mais le `.dpo` ne dit rien des filtres
   d'après le parseur. Je propose d'appliquer la règle des données : décoché, avec une ligne qui dit que l'enregistrement ne déclare
   pas ses filtres (comme un BrainVision sans `[Comment]`). C'est alors une différence voulue avec le jumeau. Autre solution : le cocher
   pour tout Curry.
3. **`tools_curry/inspect_curry.py`** (non versionné) importe `curry_header` : il ne marchera plus. Je le laisse tel quel, je l'adapte à
   `curry_lib`, ou je le supprime ?

- **Tests proposés** : quick et standard à chaque étape, full à la fin, plus la comparaison sur Curry.

**Réponses (2026-11-18)** : 1. le plus économe pour moi si le résultat est le même ; 2. la règle des données (high-pass décoché) ;
3. `inspect_curry.py` supprimé.

**Fait le 2026-11-18, précisions apportées en faisant** :
- **Validation plus économe, même résultat** : au lieu de faire tourner les jumeaux gelés pendant des heures, la lecture des jumeaux
  (`curry_header`, `curry_io`) a été comparée à celle du nouveau lecteur sur les deux enregistrements réels, en lecture seule, sans
  copie (script hors dépôt, sorties : oui/non, nombres, écarts maximaux). Identiques : canaux (44, dont 32 EEG), groupes, fréquence
  et longueur, heure de début à la milliseconde, impédances, types, positions, `meas_date`, signal (24 fenêtres de 30 s par nuit,
  écart 0), sélection de canaux, hypnogrammes, événements (569 et 131, écart de début 0). Tout ce qui suit la lecture est le code
  partagé des outils, déjà testé en EDF et BrainVision.
- **Différences voulues** : les marqueurs natifs du `.ceo` ne sont plus portés (comme BrainVision) ; le high-pass de l'outil 6 n'est
  plus coché ; l'outil 1 écrit sous les noms des autres formats ; `include` garde l'ordre du fichier.
- **L'heure de début Curry garde ses millisecondes** (comme le jumeau) : les débuts d'événements de l'export texte sont donc les mêmes
  qu'avant. BrainVision reste tronqué à la seconde.
- **Impédances** : tous les canaux EEG sont listés (un jamais mesuré est signalé), une entrée auxiliaire seulement avec une mesure (elles
  ne le sont presque jamais : sinon signalées à chaque enregistrement).
- **Outil 1** : les textes nomment les formats présents (*BrainVision*, *Curry*), donc une étude BrainVision garde ses messages ;
  colonne `format` dans `recording_notes.tsv`.
- **`recording_lib.read_acquisition`** remplace l'appel direct à `read_vhdr_acquisition` des outils 6 et 0 ; `FORMAT_NAMES` nomme le
  format dans les messages ; la liste des enregistrements garde `.edf`, puis `.vhdr`, puis `.cdt` pour un même nom.
- **Copies remplacées par des imports** : outils 2, 4, 6, 7 et `quality_lib`. L'outil 7 garde son analyseur XML (sans colonne
  Desaturation) et le `load_events` qui l'appelle ; `test_edf_io_sync.py` vérifie qu'aucune copie ne revient.
- **Tests** : `tests/curry_files.py` écrit des enregistrements Curry (MNE les ouvre) ; `test_curry_lib.py` (quick, 6 tests) et
  `test_curry_chain.py` (1 quick, 6 standard : outils 1, 4, 5, 6, 7, 0) passent.

### Phase 6 bis : bouton Stop (demandé le 2026-11-17, passée en TODO du SPEC le 2026-11-19)

Pendant un run, Voila ne traite aucun clic. La petite bibliothèque `jupyter-ui-poll` (conda-forge, installation à te demander) permet de
lire le clic sur *Stop* à chaque avancée de barre. Le participant en cours est abandonné proprement et *Skip* le reprend. Les réglages
de la page sont grisés pendant le run, pour qu'aucun ne change en cours de route. Après la phase 6, il y a moins de pages à équiper.
Plan détaillé et tests à écrire au début de la phase.

### Phase 6 ter : numérotation des dossiers de rapports (demandé le 2026-11-17, passée en TODO du SPEC le 2026-11-19)

`reports_01_inspection` … `reports_09_features_spectral`, pour retrouver d'un coup d'œil le rapport qu'on vient d'écrire. Chaque outil
renomme une fois les anciens dossiers d'une étude au Run, et les outils qui se retrouvent par le nom de dossier (8bis, 9, 0) sont
adaptés. On décidera alors si `derivatives/` est numéroté aussi. Après la phase 6, car les jumeaux Curry écrivent encore les anciens noms.

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
- **Outil 6** : ses drapeaux ne repèrent pas un canal simplement bruité (ils visent les canaux plats, morts, écrêtés ou bimodaux). Le
  critère du contrôle est montré à part depuis la phase 3 (comparaison au canal médian), sans entrer dans les drapeaux. Mesuré sur ces 3 enregistrements
  (2026-11-06) : **aucun des canaux `not connected` n'est signalé** (jusqu'à 18 000 µV d'écart-type), un seul des `Out of Range!`, et
  **9 canaux corrects sont signalés sur un enregistrement** (`flat_pct` de 3,5 à 5,6 % pour un seuil de 3,5 %, impédances bonnes, écart-type
  normal). C'était bien un effet de la fréquence d'échantillonnage : à 500 Hz, deux échantillons voisins diffèrent souvent de moins de
  0,06 µV, la tolérance du calcul. Corrigé en phase 3 (règle des 50 ms) : ces 9 canaux passent à 0 %.

## Sources sur le matériel

- [BIP2AUX Adapter, gain et fonction](https://www.brainproducts.com/?p=11597)
- [Sensor Tutorial, BIP2AUX Adapter](https://www.brainproducts.com/files/public/sensor-tutorial/Content/Topics/3.BIP2AUXAdapter/BIP2AUXAdapter.htm)
- [actiCHamp, plages d'entrée et résolutions](https://www.brainproducts.com/solutions/actichamp/)
