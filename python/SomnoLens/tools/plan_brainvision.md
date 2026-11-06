# Plan : format BrainVision (.vhdr, .vmrk, .eeg) et une seule version de chaque outil

**Statut** : à relire, version 1 du 2026-11-06. Rien n'est implémenté avant ta confirmation.
**Objectif immédiat** : pouvoir inspecter ces données et faire l'autoscoring (phases 1 à 3). Le reste de la chaîne suit
(phase 4), puis l'affichage haute densité (phase 5) et la suppression des jumeaux Curry (phase 6).
**Contexte** : le format n'était noté ni dans le SPEC ni dans ma mémoire, seulement une ligne dans
`plan_score_edf_standalone.md` (« .edf, .eeg, .cdt »). Ce plan et le SPEC le notent maintenant.
Ce fichier est versionné : il ne contient ni code participant, ni date d'enregistrement, ni nom de scoreur.

## Décisions prises (2026-11-06)

1. **Une seule version de chaque outil.** Les outils lisent `.edf` et `.vhdr` à travers un petit module de lecture, sans
   dossier de jumeaux. Les outils 0, 3bis et 5, qui n'ont pas de jumeau Curry, sont couverts du même coup.
2. **Les jumeaux Curry disparaissent à terme** par la même méthode (phase 6, réponse à ta question plus bas).
3. **Les fichiers compagnons sont renommés** : la détection des fichiers ne change pas (phase 0).
4. **Cz est recréé** avec la méthode de MNE (détails plus bas).
5. **Gain des canaux AUX** : pas de correction dans le code en v1. L'outil 1 signale le point, et une correction sans code
   existe déjà (section *Gain des canaux AUX*).
6. **Triggers ignorés pour l'instant.** Leur gestion est notée en TODO dans le SPEC, après les ondes lentes et les fuseaux.

## À confirmer : les choix que j'ai faits sans te les demander

- **a. Jumeaux Curry gelés de la phase 2 à la phase 6.** Ils restent dans leur dernière version et ne sont plus régénérés.
  Sinon il faudrait adapter les chaînes recherchées par les générateurs (166 lignes liées à l'EDF) pour les jeter ensuite.
  Conséquence : une correction d'un outil EDF n'atteint pas Curry pendant ce temps. Il vaut donc mieux ne pas laisser
  traîner la phase 6. Si tu y renonces, je remets les générateurs à jour une fois.
- **b. IO, seul canal EOG, va dans le slot `eog_left`** en attendant le slot vEOG du TODO *Tool 2: richer context channels*.
- **c. L'EOG est re-référencé aux deux mastoïdes moyennées pour les modèles.** IO est branché sur une entrée EEG, donc
  référencé à Cz, alors que les modèles attendent une dérivation de type E-M. Règle : un canal de contexte sur une entrée EEG
  (unité µV) partage la référence en ligne, un canal AUX est déjà bipolaire et n'est pas touché. L'EDF n'est pas concerné.
- **d. TP9 et TP10 servent de mastoïdes** : ils sont ajoutés en fin de `LEFT_MASTOIDS` / `RIGHT_MASTOIDS` de l'outil 3bis,
  ce qui ne change rien pour un EDF qui a M1/M2. Les dérivations restent affichées `C3-M2`.
- **e. `Respi` n'est pas donné à SleepFM** : le nom n'est pas dans ses alias et rien ne dit de quelle ceinture il s'agit.
  SleepFM tourne alors sans respiration, comme pour une étude sans ce canal.
- **f. Les noms de notebooks ne changent pas** (`1_inspect_edf_voila.ipynb`...), seuls leurs textes disent « recording ». Le
  renommage ira avec *Clean-up before distribution*.
- **g. Hors périmètre** : l'outil 1bis (voir *Limites*) et les scripts batch de l'outil 1, qui restent EDF.

## Ce que j'ai mesuré sur les 3 enregistrements

| Point | Résultat | Conséquence |
|---|---|---|
| Lecture MNE | `read_raw_brainvision` s'ouvre en moins de 2 s, 1 minute se lit en 0,05 s | rien à contourner, 4 avertissements MNE à taire à chaque lecture |
| Contenu | 2h45 à 3h16, 500 Hz, float32, 71 canaux : 62 EEG, IO, 5 doigts, EMG, ECG, Respi | montage haute densité (phase 5) |
| Échelle EEG | valeur du fichier x 0,0488281 µV, écart-type médian 20 à 36 µV | correcte, identique à la résolution annoncée par le fabricant |
| Filtres | matériel DC à 140 Hz, logiciel 15,9155 s, soit 0,01 Hz (sauf Respi), moyenne médiane 7 à 8 µV | à lire dans le `[Comment]`, MNE range 0 Hz à cause de Respi |
| Référence | Cz, absente du fichier | décision 4 |
| Hypnogrammes | 3/3 avec exactement `floor(durée / 30)` lignes, labels W, 1, 2 | règle de longueur de l'outil 7 respectée |
| Événements | format texte Compumedics UTF-16, 205/205 sur la bonne époque depuis `New Segment`, stade identique à l'hypnogramme pour 205/205 | le parseur actuel convient, heure tronquée à la seconde comme en EDF |
| Pauses | 1 seul `New Segment` par fichier | une pause réelle décalerait l'heure : à détecter |
| Sommeil | 3 à 7 % des époques, pas de N3 ni de REM | à surveiller (phase 3 et 4) |
| Impédances | dans le `[Comment]`, avec des valeurs `Out of Range!` sur plusieurs canaux | à montrer dans l'outil 1 |
| IO | inutilisable sur 1 enregistrement sur 3 (impédance hors plage, environ 1 mV) | `EOG mode = None` pour celui-ci |

**Trois pièges que les données ont révélés :**

- **TP9, TP10, FT9, FT10 ne sont pas dans `KNOWN_EEG_CHANNEL_RE`** : classés `other`, les mastoïdes seraient perdues. Le
  lecteur remplit `transducer_type` : `EEG` si le rayon de `[Coordinates]` est non nul (62 canaux, comme MNE), `EOG` pour les
  noms IO, HEOG, VEOG, vide sinon. La regex ne change pas, donc les résultats EDF non plus.
- **Le CSV du PVT commence par le nom de l'enregistrement** (`<id>_S1_..._PVT.csv`) : il apparaîtra comme faux suffixe dans
  le menu *Event CSV* de l'outil 7. C'est sans effet sur la lecture, qui ne cherche qu'à côté de l'enregistrement. À vérifier
  en phase 4.
- **`Microéveil 5 MSE`, l'événement le plus fréquent, n'a pas de règle française** (`ARO SPONT` en a une) : il faudra le choisir
  à la main dans l'outil 4, y compris s'il doit ne rien signaler.

## Gain des canaux AUX

**Mesure** : l'ECG fait 61 à 128 mV crête à crête, et l'EMG en N2 a une médiane de 1,2 à 3,5 mV. Le rythme cardiaque est
plausible.

**Le gain est-il lisible dans le `.vhdr` ?** Oui, mais le fichier ne peut dire que ce que Recorder savait. On y lit
`Ch69=EMG,,1,mV` et `Gradient 1[mV/mV]`, c'est-à-dire aucune conversion. C'est l'équivalent de la dimension et des bornes
physiques de l'EDF, avec le même angle mort : un préamplificateur externe que personne n'a déclaré n'y apparaît pas.

**Hypothèse, à confirmer côté matériel** : un adaptateur BIP2AUX de Brain Products entre les électrodes et les entrées AUX. Son
gain documenté est de 100 (± 3 %). Divisé par 100, on obtient un ECG de 0,6 à 1,3 mV et un EMG de 12 à 35 µV, deux valeurs
plausibles.

**Comment le savoir** : regarder le montage ou demander à la personne qui l'a installé (petit boîtier sur le câble AUX ?). Le
Workspace de Recorder permet de déclarer gradient et offset par canal AUX, ce qui rendrait le `.vhdr` juste pour les futurs
enregistrements.

**Ce que ça change pour l'autoscoring** (lu dans le code) :

| Modèle | Échelle de l'EMG |
|---|---|
| GSSC | aucun effet : pas d'EMG, signaux z-scorés |
| SleepFM | aucun effet : signaux standardisés |
| YASA | **effet** : `std` et `iqr` sont calculés en µV absolus. En attendant la confirmation, utiliser le choix `None` pour son EMG |

**Corriger sans code, une fois le gain confirmé** : changer la résolution dans le `.vhdr` (`Ch69=EMG,,0.01,mV`), en gardant
l'original à côté. Le signal n'est pas touché et tous les outils voient la bonne échelle. Testé sur une copie temporaire
d'un enregistrement : rapport de 100 sur EMG et ECG, de 1 sur C3, ECG à 1,36 mV. Je ne propose donc pas de champ de gain dans
l'outil 2, sauf si un besoin apparaît côté EDF.

**En v1** : l'outil 1 affiche une ligne d'information quand un canal AUX en mV a un gradient de 1 mV/mV. Elle rappelle le cas
d'un préamplificateur et la correction ci-dessus.

## Cz

`mne.add_reference_channels(raw, ['Cz'], copy=False)` est la méthode documentée par MNE (tutoriel *Setting the EEG
reference*), et réintégrer la référence en ligne avant une référence moyenne est la pratique courante. Testé sur un
enregistrement : Cz est nul et de type EEG après l'ajout, et vaut exactement `-(TP9+TP10)/2` après référence aux mastoïdes.

- **Position** : MNE laisse la position à NaN. Le lecteur la met au sommet de la sphère, comme le veut la convention
  BrainVision, sinon les topographies de l'outil 8 tomberaient.
- **Condition** : ajouté seulement si le `[Comment]` donne `Reference Channel Name` et que ce canal manque dans le fichier.
- **Outils 1 et 2** : une ligne Cz indique « online reference, not in the file ».
- **Sans re-référence** : Cz resterait plat. L'outil 2 avertit si Cz est choisi sans référence.
- **À savoir** : la référence moyenne hérite des mauvais canaux. Sur la fenêtre du test, où des canaux sont très artefactés
  (jusqu'à 3 790 µV d'écart-type), Cz a 90 µV d'écart-type, contre 20 µV avec les mastoïdes. Le choix des canaux dans les
  outils 2 et 7 garde donc toute son importance.

## Architecture

**Nouveaux fichiers et ajouts**

- `tools/brainvision_lib.py` : `read_vhdr_header`, `read_vmrk_info` (heure de début, nombre de segments), `read_impedances`,
  `read_raw_brainvision_clean`. C'est l'équivalent de `curry_header.py` + `curry_io.py`.
- `tools/recording_lib.py` : `read_header(path)`, `read_start_datetime(path)`, `read_raw(path, include=None)`. Un simple
  `if` sur l'extension. La branche EDF appelle exactement le code actuel.
- `study_files_lib.py` : `RECORDING_SUFFIXES`, `list_recordings(folder)`. Il reste sans lecture de signal.
- Rien à ajouter à `environment.yml` : MNE lit ce format sans dépendance.

**Ce que fournit chaque format**

| Besoin | EDF (inchangé) | BrainVision |
|---|---|---|
| Trouver | `.edf` | `.vhdr`, jamais `.eeg` (extension aussi utilisée par Nihon Kohden) |
| En-tête | `read_edf_header_custom` | même dictionnaire, `physical_*` et `digital_*` vides, `transducer_type` calculé |
| Fréquence | `samples_per_record / durée` | `1e6 / SamplingInterval` |
| Durée | champs de l'en-tête | taille du `.eeg` / (canaux x octets par échantillon) |
| Heure de début | octets 168 et 176 | `New Segment` du `.vmrk`, tronqué à la seconde |
| Signal | `read_raw_edf(include=)`, suffixes `-0` | `read_raw_brainvision`, `pick`, Cz, annotations retirées (les `.fif` ne portent pas les triggers) |
| Bornes physiques | oui | aucune : `bounds_pct` absent, comme en Curry |
| Plafond d'affichage | non | ±500 µV |

**La règle se déduit des données et pas du format** : pas de bornes physiques = pas de `bounds_pct` et plafond ±500 µV. Curry
en profitera sans rien de spécifique.

**Cas à traiter dans le lecteur**

- données `INT_16` ou `VECTORIZED`, virgule codée `\1`, codepage ANSI ou UTF-8 ;
- `.eeg` absent ou nommé autrement que dans `DataFile` : échec clair, l'enregistrement passe dans `failed` ;
- plusieurs `New Segment` : ⚠ dans l'outil 1, et l'heure n'est plus fiable, donc l'export texte des événements n'est pas
  converti ;
- même nom en `.edf` et en `.vhdr` : ⚠ au scan, un seul est gardé.

## Phases

Chaque phase est validée avant la suivante, avec des commits séparés. Je propose un niveau de tests à chaque fois et j'attends
ta réponse. Dans toutes les phases, **le golden EDF synthétique ne doit pas bouger**.

### Phase 0 : les fichiers (toi, ou moi avec ton accord car c'est hors du projet)

| Avant | Après |
|---|---|
| `Hypnogram_<id>_<scoreur>.txt` | `<id>_Hypnogram_<scoreur>.txt` |
| `PSG4_ScoredEvents_<id>_Export<scoreur>.txt` | `<id>_ScoredEvents_Export<scoreur>.txt` |

- **Ne jamais renommer `.vhdr`, `.vmrk`, `.eeg`** : le `.vhdr` contient les noms des deux autres.
- Mes tests travaillent sur une copie dans `%TEMP%\bvt` (4,4 Go, 176 Go libres), supprimée à la fin. Ton dossier n'est pas
  touché.
- **Pour tes propres runs** : les outils écrivent dans le dossier d'étude, et un `.fif` de l'outil 7 pèsera environ 1,4 Go par
  enregistrement. Mieux vaut un dossier d'étude hors OneDrive.

### Phase 1 : le lecteur, sans toucher aux outils

- `brainvision_lib.py`, `recording_lib.py`, `list_recordings`.
- `tests/test_brainvision_lib.py`, avec des enregistrements écrits à la main : les cas ci-dessus, Cz, et le signal relu égal au
  signal écrit.
- **Tests proposés** : quick + ces nouveaux tests.

### Phase 2 : inspection (outils 1, 2, 3, 6)

- **Outil 1** : table des canaux, résolution, filtres du `[Comment]`, impédances avec seuil en kΩ (20 par défaut, comme le
  jumeau Curry) et `Out of Range!` signalé, pauses, ligne sur le gain AUX. L'anonymisation et la plage dynamique affichent
  « not applicable ».
- **Outil 2** : scan par `read_header`, Cz, détection du contexte (IO, EMG, ECG), aperçu par `read_raw`.
- **Outil 3** : découverte des enregistrements seulement.
- **Outil 6** : bloc de lecture, règle des bornes physiques.
- **Tests** : `tests/synthetic.py` écrit aussi un jeu BrainVision. Test d'équivalence : le même signal en EDF et en BrainVision
  donne les mêmes métriques dans l'outil 6, à la tolérance près. Puis lancement sur les vraies données en `%TEMP%`.
- **Tests proposés** : standard sur `tool1`, `tool2`, `tool3`, `tool6`.

### Phase 3 : autoscoring (outil 3bis). **Objectif atteint, arrêt pour ton test**

- `autoscoring_lib.py` : les 5 `read_raw_edf` passent par `read_raw`.
- Référence de l'EOG (choix c), dans les trois chemins : GSSC (`load_eog`), YASA, SleepFM.
- Mastoïdes TP9 et TP10 (choix d).
- Section 4 sur les vraies données, contre ton scoring expert.
- **Fiabilité à juger avec cette section** : les modèles ont appris sur des nuits entières, ici 93 à 97 % d'éveil pendant une
  tâche, c'est une question scientifique que le format ne règle pas.
- **Tests proposés** : standard sur `tool3bis`, puis `tests/score_edf_equivalence.py`, car la Section 4 ne doit pas changer.
- **Fin de phase** : tu testes dans Voila.

### Phase 4 : le reste de la chaîne (outils 4, 5, 7, 0)

- **Outil 4** : découverte, événements, heure de début.
- **Outil 5** : `read_edf_header_info` et les lectures `include=[canal]` passent par le lecteur.
- **Outil 7** : les 5 sites de découverte, les blocs `[load]` et `[H]`, l'heure de début, la référence de l'EOG dans le compagnon.
- **Outil 0** : par `edf_io_lib`.
- **Vérifier sans modifier** : 5bis, 8, 8bis, 9 sur des sorties BrainVision.
- **Points à regarder sur ces données** :
  - l'outil 8bis prend N2 comme stade de référence, soit 4 à 20 époques seulement selon l'enregistrement, ce qui est peu pour
    juger un canal ;
  - les métriques de l'outil 5 sur des siestes en journée, et ses *lights off / on* ;
  - la mémoire de l'outil 7 : environ 2,9 Go par enregistrement avant ses copies, sans problème sur 31 Go.
- **Tests proposés** : full, une fois.

### Phase 5 : affichage haute densité

Le principe reste : **adapté au nombre de canaux, jamais au format, inerte à 12 canaux ou moins** (`HD_CHANNEL_THRESHOLD`), donc
EDF inchangé. Audit à 62 canaux :

| Outil | État lu dans le code | À faire |
|---|---|---|
| 8, 8bis | adaptés dans `qc_rejected_epochs_lib` | vérifier à 62 canaux (Curry allait jusqu'à 44) |
| 9 | **une rangée de n colonnes, largeur plafonnée à 20 pouces** (`_fig_width`), soit 0,3 pouce par canal | à refaire (grille ou sélection de canaux), pas d'usage connu en haute densité |
| 6 | grilles `5 x ncols` par `3,5 x nrows`, superposition déjà transparente | lisibilité et poids du rapport HTML |
| 7 | heatmap et rapport | vérifier |
| 1, 2 | tables et listes de 71 canaux | vérifier |
| 0 | navigateur et explorateur | vérifier, pas d'usage connu en haute densité |
| 3bis, 5 | quelques canaux seulement | rien |

- **Tests proposés** : quick, plus un test qui prouve que rien ne change à 12 canaux ou moins.

### Phase 6 : suppression des jumeaux Curry

**Réponse à ta question : oui, c'est faisable, et je le recommande.** D'après le SPEC, ce que font les jumeaux tient en peu de
choses :

| Outil | Ce que fait le jumeau | Dans la version unique |
|---|---|---|
| 1 | page dédiée, contrôle des impédances | déjà là en phase 2 ; sorties dans `reports_inspection/` et plus `reports_inspection_curry/` |
| 2 | masque EEG par groupe du `.dpo` | `transducer_type` rempli depuis ce groupe |
| 3 | découverte | rien de plus |
| 4 | une seule source d'événements, pas de contrôle croisé | même chaîne TXT, contrôle montré seulement avec deux sources |
| 6 | sans `bounds_pct`, plafond ±500 µV | déjà la règle de phase 2 |
| 7 | lecture Curry, heure du `.dpo` | par le lecteur |
| 8, 8bis | titre et chemin d'import seulement | jumeaux supprimés, rien à porter |

**Ce qui en découle**

- `curry_header.py` et `curry_io.py` deviennent `tools/curry_lib.py`, branché dans `recording_lib.py`.
- Les outils 0, 3bis, 5 fonctionnent aussi sur Curry. Deux TODO se referment : *A Curry twin of tool 0* et *EDF reading of tools
  2, 4, 6, 7 through `edf_io_lib`*. Plus aucun générateur ne réécrit ces blocs : les copies peuvent disparaître et
  `test_edf_io_sync.py` se réduit.
- **Supprimés** : 8 notebooks, 7 générateurs, `test_curry_twins.py`. `test_notebooks_execute.py` et `test_ui_lib.py` sont
  adaptés.
- `tools_curry/inspect_curry.py` n'est pas versionné : je te le demande avant d'y toucher.

**Validation** : c'est la seule phase où l'on peut perdre quelque chose, mais les 2 enregistrements de `tools/test_data_curry/`
(7 Go chacun) permettent de la vérifier.

1. Faire tourner les jumeaux actuels et sauvegarder toutes les sorties, avant toute suppression.
2. Faire tourner les outils uniques sur les mêmes données.
3. Comparer tous les fichiers (`tests/refactor_check.py`) : identiques, sauf différences listées et justifiées. Run long, avec
   keep-awake.
4. Tu vérifies dans Voila sur Curry, comme le SPEC le demandait déjà.

- **Tests proposés** : full, plus cette comparaison.

### Phase 7 : documentation et fin

- **SPEC** : section *Recording formats* (EDF, BrainVision, Curry) à la place de *Curry 9 support*, structure du dépôt, TODO
  fermés, ordre des TODO.
- **CLAUDE.md** : la partie *Curry twins* se réduit fortement, et une règle courte s'ajoute (toute lecture passe par
  `recording_lib`).
- **README et TESTS.md** à jour.
- Ce plan est supprimé en commit séparé.
- Les modifications du SPEC te sont proposées à chaque phase.

## Limites connues

- **Outil 1bis** : il reste EDF, car le `.vhdr` n'a pas de champ patient. Mais le `.vmrk` porte la date d'enregistrement et le
  `[Comment]` le numéro de série de l'ampli : avant tout partage de données, il faudra une anonymisation dédiée. À noter en TODO si
  tu le souhaites.
- **Données compressées ou segmentées** (exports Analyzer) : non couvertes, seul l'enregistrement continu l'est.
- **Triggers** : ignorés, voir SPEC *Task triggers*.
- **Doigts** : les 5 canaux restent inutilisés. Ils relèvent du TODO *Tool 2: richer context channels* (plusieurs EMG).

## Sources sur le matériel

- [BIP2AUX Adapter, gain et fonction](https://www.brainproducts.com/?p=11597)
- [Sensor Tutorial, BIP2AUX Adapter](https://www.brainproducts.com/files/public/sensor-tutorial/Content/Topics/3.BIP2AUXAdapter/BIP2AUXAdapter.htm)
- [actiCHamp, plages d'entrée et résolutions](https://www.brainproducts.com/solutions/actichamp/)
