# Plan : sélection automatique des fichiers (hypnogrammes, entrées, dossiers)

Statut : brouillon à annoter, rien n'est implémenté (2026-11-03). Remplace, une fois fait, les deux points de SPEC
*Planned modules* « Companion suffixes chosen from the detected ones, in the batch tools » et « File selection
pointed at the expected files ».

## 1. Objectif

Au début, chaque sélecteur faisait choisir chaque fichier, pour que l'utilisateur comprenne la logique des outils.
Avec le nombre de fichiers et de rapports produits, c'est devenu lourd, et le pipeline s'adresse d'abord à un public
débutant (une version expert pourra venir plus tard). Le but : **le dossier d'étude est la seule chose à choisir**.
Chaque outil trouve le reste, dit ce qu'il a trouvé et ce qui manque, et suit les noms que l'utilisateur a choisis
dans les outils précédents.

Trois défauts à corriger :
1. **Le choix automatique prend les hypnogrammes produits par la toolbox** (signalé le 2026-11-02). L'outil 3 prend
   `_hypnogram_gssc.txt` (19 caractères) au lieu de `_Hypnogram_Export.txt` (21), les outils 5, 6, 7 et 0 prennent
   `_Hypnogram_remapped_rescored.txt` au lieu de `_Hypnogram_remapped.txt`.
2. **Le comptage des suffixes confond les participants dont le nom commence pareil.** Un fichier est compté pour
   tout enregistrement dont le nom le préfixe, dans n'importe quel dossier : `10_Hypnogram_remapped.txt` est
   compté pour l'enregistrement `1`, avec le suffixe `0_Hypnogram_remapped.txt`. Avec les participants 1, 2, 10 et
   20, ce faux suffixe compte 2 sur 4, passe le seuil de 50 % et gagne parce qu'il est plus long.
3. **La lourdeur des sélecteurs** : dans l'outil 7, les sélecteurs de la configuration de l'outil 2, de
   `event_remap.json` et de `quality_summary.tsv` s'ouvrent seulement dans le dossier d'étude, il faut encore
   naviguer et sélectionner. Les outils 8, 8bis et 9 font choisir deux ou trois dossiers à la main.

## 2. Décisions prises (2026-11-03)

- Un plan, trois phases, un commit par phase.
- Une liste « Inputs found » : le dossier d'étude est le seul sélecteur, chaque entrée est trouvée et affichée, un
  bouton « Choose another… » ouvre un sélecteur seulement si besoin.
- **Un suivi des noms choisis par l'utilisateur** : si l'utilisateur a changé un nom par défaut dans un outil (par
  exemple le suffixe des hypnogrammes remappés), la sélection automatique des outils suivants le suit (§4.2).
- Runs versionnés (outils 8, 8bis, 9) : choisis d'office quand il n'y en a qu'un, un menu quand il y en a plusieurs.
- Les champs qu'un débutant n'a pas à toucher vont dans un encadré « Advanced » replié. La liste est à redéfinir
  ensemble (§6.4).

## 3. État actuel (inventaire)

| Outil | Sélecteurs et champs aujourd'hui | Déjà automatique |
|---|---|---|
| 0 | dossier d'étude, participant, menu des hypnogrammes + « Other file… » | configuration de l'outil 2, `event_remap.json`, `quality_summary.tsv`, `participants.tsv` |
| 1 | dossier d'étude | |
| 2 | dossier d'étude, nom du JSON à l'enregistrement | |
| 3 | dossier d'étude, suffixe d'entrée, suffixe de sortie | suffixe d'entrée pré-rempli |
| 3bis | dossier d'étude, suffixe expert, trois suffixes de sortie | configuration de l'outil 2, suffixe expert pré-rempli (écarte déjà ses propres sorties) |
| 4 | dossier d'étude, suffixes TXT et CSV, nom de `event_remap.json` | suffixes pré-remplis |
| 5 | dossier d'étude, dossier de sortie, event remap, participant info, quatre suffixes, custom stages | `event_remap.json` pré-sélectionné, suffixes pré-remplis |
| 5bis | dossier d'étude, ou la table directement | |
| 6 | dossier d'étude, suffixe hypnogramme, custom stages, configuration de l'outil 2 (à sélectionner) | suffixe pré-rempli |
| 7 | dossier d'étude, quality summary, configuration de l'outil 2, event remap (trois sélecteurs à parcourir), trois suffixes, custom stages, dossier de sortie | suffixes pré-remplis |
| 8 | dossier d'étude (optionnel), dossier d'époques, dossier de rapports (optionnel), participant | sélecteurs ouverts dans `derivatives/` |
| 8bis | dossier d'étude (optionnel), dossier d'époques, dossier de rapports | idem |
| 9 | dossier d'étude (optionnel), dossier d'époques, dossier de décision (optionnel), participant info | idem |

La règle de choix de l'hypnogramme est recopiée dans six outils (0, 3, 3bis, 5, 6, 7), avec des variantes : 3bis
écarte ses sorties, 5 et 3bis écartent `summary` / `studylog`, 6 et 7 seulement `event`.

## 4. Phase 1 : le bon hypnogramme

### 4.1 Comptage corrigé

`companion_suffix_counts(recordings, files)` : un fichier n'est compté que **dans le dossier de son enregistrement**
(les outils lisent `<dossier de l'enregistrement>/<nom><suffixe>`), et pour l'enregistrement **au nom le plus long**
qui le préfixe (`10_Hypnogram_remapped.txt` va à `10`, jamais à `1`). Seul le nom compte, donc la fonction sert
aussi aux jumeaux Curry (`.cdt`).

### 4.2 Le registre des noms : `config_param/file_names.json`

Un petit fichier de l'étude où chaque outil note les noms qu'il a écrits (ou lus). La sélection automatique le
consulte en premier. C'est plus sûr qu'une liste de motifs : un nom personnalisé est suivi, et quand une étude a
l'ancien et le nouveau nom (outil 3 relancé), le registre dit lequel est le bon.

```json
{
  "hypnogram_scored":   {"value": "_Hypnogram_Export.txt", "tool": "3", "saved_at": "2026-11-03 10:12"},
  "hypnogram_remapped": {"value": "_hypno_remapped.txt",   "tool": "3", "saved_at": "2026-11-03 10:12"},
  "hypnogram_gssc":     {"value": "_hypno_gssc.txt",       "tool": "3bis", "saved_at": "..."},
  "hypnogram_rescored": {"value": ["_hypno_remapped_rescored.txt"], "tool": "0", "saved_at": "..."}
}
```

Règles (celles du registre des exclusions) : chaque outil réécrit **seulement ses propres clés** ; fichier absent =
rien d'enregistré, sans avertissement ; fichier illisible = ⚠, règles de secours, et il n'est **jamais écrasé** ;
écriture atomique ; valeurs identiques = fichier inchangé. Une clé est écrite quand l'outil écrit vraiment ses
fichiers (Save, Run), jamais au Scan.

Clés de la phase 1 : `hypnogram_scored` et `hypnogram_remapped` (outil 3 : le suffixe lu, le suffixe écrit),
`hypnogram_gssc` / `hypnogram_yasa` / `hypnogram_sleepfm` (outil 3bis), `hypnogram_rescored` (outil 0, la liste des
suffixes écrits). Clés de la phase 3 : `channel_config` (outil 2), `event_remap`, `events_txt`, `events_csv`
(outil 4), `output_root_5`, `output_root_7` (dossiers de sortie avancés des outils 5 et 7).

### 4.3 Une seule règle : `pick_hypno_suffix(counts, intent, names, own_outputs=())`

Deux intentions : `scored` (outil 3 : l'export brut du logiciel de scorage) et `expert` (outils 0, 3bis, 5, 6, 7 :
l'hypnogramme remappé par l'outil 3). Ordre :
1. **Le nom enregistré** pour cette intention, s'il existe pour au moins un enregistrement.
2. Sinon, **la règle actuelle**, sur les candidats restants après avoir écarté :
   - les exports d'événements, `summary` et `studylog` (comme aujourd'hui) ;
   - les hypnogrammes de la toolbox : les noms enregistrés par 3bis et 0, et les noms par défaut, anciens et
     nouveaux (`_hypno_<algo>.txt`, `_hypnogram_<algo>.txt`, tout suffixe finissant par `_rescored.txt`) ;
   - les sorties de l'outil lui-même (`own_outputs` : les suffixes de sortie de 3bis, celui de l'outil 3) ;
   - pour `scored` : les remappés (nom enregistré, `_hypno_remapped.txt`, `_Hypnogram_remapped.txt`) ;
   - pour `expert` : l'export brut enregistré par l'outil 3.

   Puis `expert` = le plus long parmi ceux présents pour au moins 50 % du maximum, `scored` = le plus fréquent, le
   plus court à égalité.
3. **S'il ne reste rien** : aucun choix automatique, message orange (« no expert hypnogram detected: choose one in
   the list »). Tous les suffixes restent affichés avec leur nombre de fichiers, l'utilisateur peut donc choisir à
   la main un hypnogramme de la toolbox (par exemple faire tourner la chaîne sur GSSC).

La fonction rend `(suffixe, nombre, raison)`. La raison est affichée : « recorded by tool 3 » ou « detected ».

### 4.4 Noms courts, pour les nouvelles études seulement

Nouveaux défauts : outil 3 `_hypno_remapped.txt`, outil 3bis `_hypno_gssc.txt`, `_hypno_yasa.txt`,
`_hypno_sleepfm.txt`, et les défauts de lecture des outils 0, 3bis, 5, 6, 7 suivent. **Une étude garde ses noms** :
le champ de sortie de l'outil 3 prend le nom enregistré, sinon un suffixe remappé déjà présent (l'ancien défaut),
sinon le nouveau. Même règle pour les trois sorties de 3bis. Sans cela, relancer l'outil 3 sur une étude déjà
traitée écrirait une seconde copie remappée sous le nouveau nom. Le fichier recorrigé de l'outil 0 garde sa règle
(`<nom de l'hypnogramme>_rescored.txt`).

### 4.5 Ce qui change

- **Nouvelle bibliothèque `tools/study_files_lib.py`** : le registre des noms, le comptage, les règles de choix (et
  en phase 3 la localisation des entrées). Pourquoi pas `edf_io_lib` : ces fonctions ne lisent aucun EDF, et les
  notebooks des outils 5 à 9 et les jumeaux Curry 4, 6, 7 n'importent pas `edf_io_lib`. Ils importent tous
  `ui_lib` avec la même recherche du dossier `tools/`, donc la nouvelle bibliothèque se charge de la même façon,
  sans MNE. `companion_suffix_counts`, `select_hypno_suffix` et `select_event_suffix` y passent depuis `edf_io_lib`
  (seul l'outil 0 et `tests/test_edf_io_lib.py` les utilisent).
- **Outils 0, 3, 3bis, 5, 6, 7** : le comptage et la règle recopiés sont remplacés par les appels à la
  bibliothèque. L'affichage ne change pas (tous les suffixes avec leur nombre, `← selected`), plus la raison.
- **Outils 3, 3bis, 0** écrivent leurs clés du registre.
- **Jumeaux Curry 3, 6, 7** : relancer les générateurs, et mettre à jour leurs chaînes de correspondance là où elles
  visent les lignes remplacées (`_make_tool7_curry.py` en a plusieurs dans ce bloc).
- **SPEC** : *Cross-cutting → Hypnogram-suffix auto-detection* réécrit, *Study folder layout* (nouveaux noms), §0,
  §3, §3bis (corriger au passage `_Hypnogram_gssc.txt`, alors que le code écrit `_hypnogram_gssc.txt`), §5, §6, §7.
  **CLAUDE.md** : un rappel court (une seule règle de choix dans `study_files_lib`, le registre des noms).

### 4.6 Tests

- Tests unitaires de la bibliothèque (quick) : participants 1, 2, 10, 20 ; sorties de 3bis et de 0 présentes (le cas
  du 2026-11-02) ; nom personnalisé suivi grâce au registre ; ancien et nouveau nom présents ; registre illisible.
- Un test de notebooks sur une copie : une étude avec `_Hypnogram_Export.txt`, `_Hypnogram_remapped.txt`,
  `_hypnogram_gssc.txt` et `_Hypnogram_remapped_rescored.txt` : l'outil 3 prend l'export, les outils 5, 6, 7
  prennent le remappé.
- La chaîne : les données synthétiques gardent leurs noms actuels (l'ancien défaut), ce qui prouve la compatibilité
  et laisse les goldens inchangés. `tests/refactor_check.py` avant / après sur la chaîne et sur `outside_chain.py 3
  3bis`. Seul changement attendu : les noms de sortie par défaut de 3bis (`outside_chain 3bis`, test 3bis de
  `test_tool0.py`).
- Niveau proposé : quick + standard sur `tool0`, `tool3`, `tool3bis`, `tool5` à `tool9` (la chaîne repart de
  l'outil 5, dont le notebook change).

## 5. Phase 2 : menus déroulants des suffixes

- Chaque champ de suffixe (hypnogramme, événements TXT, événements CSV, summary) devient un **menu des suffixes
  détectés avec leur nombre de fichiers** (`_hypno_remapped.txt (40/40)`), pré-sélectionné par la règle de la phase
  1, plus une dernière option **« Other suffix… »** qui fait apparaître le champ texte.
- Sous chaque menu, la **couverture** : `38/40 recordings (missing: S12, S27)` (au plus 10 noms, puis « … »). C'est
  l'avertissement « fichier manquant » pour les compagnons.
- Le champ texte reste la valeur lue par le run et **garde son nom** (`hypno_suffix`, `suffix_box`,
  `txt_hypno_suffix`, `txt_suffix`, `csv_suffix`, `txt_summary_suffix`, …) : les tests qui le règlent ne changent pas.
  Le menu écrit dans le champ.
- Le menu est construit par `ui_lib` (la seule bibliothèque qui construit des widgets) : `UI.suffix_choice(...)`.
- Outils 3, 3bis, 4, 5, 6, 7. L'outil 0 a déjà son menu : il passe par le même constructeur. Jumeaux Curry 3, 4, 6, 7.
- Tests : quick (le constructeur) + un test de notebook « Other suffix… ». Niveau proposé : standard sur les outils
  concernés.

## 6. Phase 3 : liste « Inputs found », runs versionnés, encadré « Advanced »

### 6.1 Principe

La section 1 = le dossier d'étude (seul sélecteur) + une liste construite dès qu'il est choisi. Une ligne par entrée :
- ✅ trouvée : le chemin relatif à l'étude, et d'où vient le choix (« recorded by tool 2 », « default location ») ;
- ❌ obligatoire et absente : l'outil qui la produit (« run tool 2 first »), le Run refuse avec ce message ;
- ℹ️ optionnelle et absente : ce qui est désactivé sans elle (« event flagging off ») ;
- un bouton **« Choose another… »** qui fait apparaître le sélecteur existant. Les sélecteurs gardent leurs noms
  (`fc_config`, `fc_quality`, `fc_events`, `fc_remap`, `fc_subj`, `fc_table`, `fc_raw`, `fc_reports`,
  `fc_decision`) : les tests qui les pilotent continuent de marcher. Un fichier choisi à la main l'emporte, la ligne
  le dit (« chosen by hand »).

Les menus de suffixes de la phase 2 deviennent des lignes de cette liste. Les custom stages y figurent en lecture
seule (`config_param/custom_stages.json`: N4), le champ de remplacement passe dans l'encadré « Advanced ».

### 6.2 Ce que chaque outil trouve

| Outil | Entrée | Où elle est cherchée | Si absente |
|---|---|---|---|
| 0 | tout ce qu'il trouve déjà | passe par la bibliothèque et le registre | inchangé |
| 3bis, 6, 7 | configuration de l'outil 2 | nom enregistré par l'outil 2, sinon `config_param/remap_reref_persubject.json` | ❌ |
| 4 | mapping existant | nom enregistré par l'outil 4, sinon `config_param/event_remap.json` | ℹ️ nouveau mapping |
| 5, 7 | event remap | idem | ℹ️ (5 : événements non comptés, 7 : event flagging off) |
| 5, 9 | participant info | `participants.tsv` / `.csv` / `.xlsx` à la racine de l'étude (comme l'outil 0) | ℹ️ |
| 5bis | `global_sleep_metrics.tsv` | `reports_features_macrostructure/` dans l'étude, ou sous le dossier de sortie enregistré par l'outil 5 | ❌ « run tool 5 » |
| 7 | quality summary | `reports_quality_overview/quality_summary.tsv` | ⚠️ recommandé (sans lui, aucun canal pré-désélectionné) |
| 8, 8bis, 9 | époques, rapports, décision | §6.3 | époques ❌ ; rapports : 8 ℹ️ (flags recalculés), 8bis ❌ ; décision (9) ℹ️ |

Plusieurs configurations de l'outil 2 (`remap_reref_persubject*.json`) : un menu, la dernière enregistrée par
l'outil 2 en premier. Garder séparées les sorties des outils suivants reste hors périmètre (§7).

### 6.3 Runs versionnés (outils 8, 8bis, 9)

- Le dossier d'étude devient obligatoire (aujourd'hui optionnel) : c'est l'ancre de la recherche.
- **Dossiers d'époques** : les dossiers `derivatives/preproc_epo*` qui contiennent des `*_all-epo.fif` (et ceux sous
  le dossier de sortie enregistré par l'outil 7). Un seul : choisi d'office. Plusieurs : un menu avec le nombre de
  participants de chacun. Plus « Other folder… » (le sélecteur actuel) pour un run hors de l'étude, où le calcul
  actuel du dossier d'étude depuis `derivatives/` est gardé.
- **Rapports** : appariés par le nom (`preproc_epo_v2` → `reports_preproc_v2`). Pas de dossier apparié : ⚠ et
  « Choose another… ».
- **Décision (outil 9)** : un menu `rejection_auto*` / `rejection_manual*` / « none: keep every epoch ». Le choix par
  défaut quand les deux existent est un point ouvert (§8).

### 6.4 Encadré « Advanced » : proposition à redéfinir ensemble

| Outil | Proposé dans « Advanced » | Reste visible |
|---|---|---|
| 2 | nom du fichier JSON enregistré | sélection et remap des canaux |
| 3 | suffixe de sortie | hypnogramme brut (menu), sections suivantes |
| 3bis | les trois suffixes de sortie | modèles, hypnogramme expert (menu) |
| 4 | nom de `event_remap.json` | suffixes TXT / CSV (menus) |
| 5 | dossier de sortie, champ custom stages | liste des entrées, suffixes (menus) |
| 6 | champ custom stages | liste des entrées |
| 7 | dossier de sortie, champ custom stages | liste des entrées, case « manual annotations » |
| 0, 9 | champ custom stages | liste des entrées |

Un encadré « Advanced » ouvert et modifié s'affiche avec un petit ⚠ (« default changed ») pour ne pas l'oublier.

### 6.5 Ce qui change

- `study_files_lib` : les fonctions de localisation (une par entrée) et l'écriture des clés de la phase 3.
- `ui_lib` : la ligne de la liste (`UI.input_row(...)`) et l'encadré (`UI.advanced_box(...)`).
- Outils 0, 2, 3, 3bis, 4, 5, 5bis, 6, 7, 8, 8bis, 9 ; jumeaux Curry 2, 3, 4, 6, 7, 8, 8bis (générateurs relancés).
- SPEC : *Study folder layout*, *Voila page conventions* (la liste et l'encadré), la section 1 de chaque outil.
- Tests : les sélections de `tests/pipeline.py` sont gardées (elles testent « Choose another… ») ; un nouveau test
  par outil avec seulement le dossier d'étude (tout trouvé) doit donner les mêmes sorties que la chaîne. Niveau
  proposé : full à la fin de la phase.
- Si la phase est trop grosse pour un commit : 3a (outils 0, 2, 3, 3bis, 4, 5, 5bis, 6, 7), 3b (8, 8bis, 9).

## 7. Hors périmètre

- **Séparer les sorties** de plusieurs configurations de l'outil 2 ou de runs versionnés (TODO *Renaming the
  outputs*). Trouvé pendant l'inventaire : l'outil 8bis écrit ses décisions dans `derivatives/rejection_auto/` quel
  que soit le dossier d'époques, donc `preproc_epo` et `preproc_epo_v2` partagent un seul dossier de décision. À
  ajouter à ce TODO.
- L'audit de l'organisation de la base par l'outil 1 (la couverture de la phase 2 en couvre la partie « compagnon
  manquant »).
- Le découpage des outils 1 à 4 en une cellule par section (les phases 2 et 3 modifient les outils 2, 3, 4 dans leur
  cellule unique : faisable, juste plus lent).
- La lecture EDF des outils 2, 4, 6, 7 par `edf_io_lib` (test Curry).

## 8. Points ouverts (à annoter)

1. Le nom du registre : `config_param/file_names.json` ?
2. Seuls des hypnogrammes de la toolbox présents : aucun choix automatique + message orange (recommandé), ou les
   choisir avec un avertissement ?
3. Outil 9, les deux dossiers de décision présents : pré-choisir le plus récent (recommandé), toujours le manuel
   (outil 8), toujours l'automatique (8bis), ou aucun (l'utilisateur choisit) ?
4. Entrée obligatoire absente : le Run refuse avec le message (recommandé), ou il prévient seulement ?
5. Données de test synthétiques : garder les anciens noms (recommandé : prouve la compatibilité, goldens inchangés),
   les nouveaux noms étant testés à part ?
6. Le tableau « Advanced » (§6.4).
7. Les tutoriels (`.docx`) citent-ils les noms d'hypnogrammes ? Si oui, tu les mets à jour.

## 9. Ordre

Phase 1 → commit, phase 2 → commit, phase 3 (ou 3a, 3b) → commit. À la fin : retirer les deux points de SPEC
*Planned modules*, ajouter la collision de 8bis au TODO *Renaming the outputs*, supprimer ce plan (commit à part).
