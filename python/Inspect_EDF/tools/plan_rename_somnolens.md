# Plan : renommer `Inspect_EDF` en `SomnoLens` (+ remise à jour des mentions)

Statut : **EN COURS**, lancé le 2026-10-29. Ordre réel : les phases 2 et 3 (textes) sont faites et mises en scène
**avant** le renommage, car VS Code (et la session Claude) tient le dossier ouvert : le `git mv` ne peut se faire
qu'en dernier, VS Code fermé. Reste : commit des textes, Phases 0-1 (toi, hors VS Code), puis Phases 4-5 dans une
nouvelle session Claude ouverte sur `SomnoLens/`.

## Contexte

- Le dossier `Dream-Toolkit/python/Inspect_EDF/` porte le nom de son premier outil, alors que le pipeline
  couvre maintenant inspection, harmonisation, prétraitement et extraction de mesures, en EDF et Curry.
- Nouveau nom retenu : **SomnoLens**. Vérifié le 2026-10-14 : aucun outil scientifique de ce nom (GitHub,
  PyPI, npm, web). somnolens.com est pris par un site de photographie, et « somnolens » est un mot courant
  (somnolence en danois/suédois/norvégien) : toujours écrire « SomnoLens » + « EEG » ou « sleep ».
- Le dépôt git commence à `Dream-Toolkit/` : le renommage est un `git mv` d'un sous-dossier, l'historique
  est conservé (`git log --follow`).

## Décisions (prises le 2026-10-16)

| # | Question | Décision |
|---|---|---|
| D1 | Renommer l'environnement conda `inspect_edf` | **Oui** → `somnolens` |
| D2 | Casse | **`SomnoLens`** (dossier, titres), **`somnolens`** (environnement, futur paquet) |
| D3 | Fichiers de l'outil 1 (`1_inspect_edf_*`, `1_inspect_curry_voila`) | **Gardés** : le nom décrit bien l'outil |
| D4 | `README.docx` | **Réécrit en `README.md`** (anglais), fonctionnalités actuelles **et futures**, installation légère / complète (voir Phase 6) ; le .docx est supprimé |
| D5 | `README.md` racine de `Dream-Toolkit/` | **Pas réécrit** : on y ajoute seulement un pointeur vers `python/SomnoLens/README.md` |
| D6 | Brouillons non versionnés à chemins absolus (`tools/test_Savitzky-Golay/diagnostics/*.py`) | **Pas touchés** |
| D7 | Fichiers non versionnés (`6bis_metrics_correlation.py`, `inspect_curry.py`, `test_mne_curry_header.py`, `test_data_curry/`, `scientific_com/`) | **Ni versionnés ni documentés** |

## TODO

### Phase 0 : préparation (faite par toi, après les phases 2-3)
- [x] Liste envoyée au lancement (2026-10-29).
- [ ] État git propre : commiter les textes mis en scène (commit 1, message proposé dans la conversation), et
      commiter ou mettre de côté ton travail en cours (`tools/1_inspect_edf_voila.ipynb`, tutoriels `.docx`).
      Le renommage reste alors un commit **pur** (100 % de similarité, `git log --follow` sans surprise).
- [ ] Fermer tout ce qui tient des fichiers ouverts dans le dossier : **toutes** les fenêtres VS Code (donc cette
      session Claude), serveurs Voila/Jupyter, noyaux Python, Word (tutoriels), PowerPoint (poster), explorateur
      Windows ouvert dans le dossier. Sous OneDrive, un fichier verrouillé fait échouer le renommage à moitié.
- [ ] (Conseillé) Suspendre la synchronisation OneDrive le temps du renommage (icône OneDrive → Suspendre 2 h).

### Phase 1 : renommer (toi, dans une fenêtre PowerShell ouverte hors de VS Code)
```powershell
cd "C:\Users\yvan.nedelec\OneDrive - ICM\Documents\RE_yvan\projects\Dream-Toolkit"
git mv python/Inspect_EDF python/SomnoLens
git status                                    # que des "renamed:" (pas de deleted + new file)
& "$env:LOCALAPPDATA\miniforge3\Scripts\conda.exe" rename -n inspect_edf somnolens
& "$env:LOCALAPPDATA\miniforge3\envs\somnolens\python.exe" -c "import mne, yasa, gssc; print('ok')"
```
- [ ] `git mv` : `git status` doit montrer des *renamed* (pas de suppressions + ajouts). Les fichiers non
      versionnés (test_data, etc.) suivent le dossier. Si `git mv` échoue (« Permission denied ») : un fichier est
      encore ouvert, fermer le programme et relancer (rien n'a bougé).
- [ ] `conda rename` (pas de réseau nécessaire ; il copie l'environnement puis supprime l'ancien, quelques minutes), puis
      vérifier l'import. Ton PATH utilisateur ne contient aucun dossier `inspect_edf` (vérifié) : rien à y changer.
- [ ] Commit 2 (le renommage seul), puis rouvrir VS Code sur `python/SomnoLens/` et lancer une nouvelle session
      Claude : « continue le plan_rename_somnolens, phases 4-5 ».

### Phase 2 : mentions actives à mettre à jour (re-grepées le 2026-10-29, faites et mises en scène le 2026-10-29)
Dossier (`Inspect_EDF`) :
- [x] `CLAUDE.md` : titre, « from `Inspect_EDF/` » (section *Canonical launch cwd*).
- [x] `SPEC.md` : titre, racine de l'arborescence, « repo root (`Inspect_EDF/`) », titre *Code review … findings*.
- [x] `tests/nbdriver.py` (commentaire de `REPO_ROOT`).
- [x] `tools_curry/_make_tool{2,3,4,6,7}_curry.py` : docstring « Run from Inspect_EDF root ».
- [x] `tools/plan_score_edf_standalone.md` (trouvé au re-grep).

Environnement (`inspect_edf` → `somnolens`) :
- [x] `environment.yml` (`name:`).
- [x] `CLAUDE.md` (*Running Python*) ; `SPEC.md` (env, procédure notebooks, gssc, *How to run*, tests, §3bis, §5,
      *Lighter conda environments*).
- [x] `tools/autoscoring_lib.py` : **message affiché à l'utilisateur** quand GSSC manque.
- [x] Docstrings de lancement : `tests/conftest.py`, `tests/make_golden.py`, `tests/refactor_check.py`,
      `tools/6bis_metrics_correlation.py`, `tools/compare_flat_spectrogram_fix.py`,
      `tools/simple_hypnospectro_yasa_vs_fix.py`, `tools/images/_make_unicorn_gif.py`,
      `tools_curry/_make_tool*_curry.py` (7 fichiers), `tools_curry/test_mne_curry_header.py`,
      `tools/test_data/test_spectral/make_test_spectral.py` (non versionnés : modifiés, pas mis en scène).
- [x] `tools/plan_3bis_autoscoring.md` (plan gardé en référence).
- [x] Trouvé au re-grep : métadonnées `kernelspec` `inspect_edf` (pas un noyau Jupyter enregistré, Voila
      retombait sur `python3`) → `python3` dans 3bis, 7, Curry 1, Curry 7 ; texte « Environment » de 3bis et
      `CLAUDE.md` §3bis : SleepFM tourne aussi dans `somnolens` (ils disaient encore « needs the `autoscore` env »).
- [ ] **Reste** : `"display_name": "inspect_edf"` dans `tools/1_inspect_edf_voila.ipynb` (métadonnée, son
      `name` est déjà `python3`). Non touché : ce fichier porte ton travail en cours. À changer en
      `"Python 3 (ipykernel)"` avec ton commit, ou VS Code le réécrira tout seul en `somnolens` à la prochaine
      sauvegarde avec ce noyau.

Documents :
- [x] (D4) `README.md` de SomnoLens (remplace `README.docx`, supprimé par `git rm`) : présentation, outils
      actuels (0 à 9, Curry expérimental), outils prévus, installation (actuelle + légère / complète prévue),
      démarrage rapide, dossier d'étude, licence (à confirmer, GSSC AGPL, SleepFM CC BY-NC).
- [x] (D5) Un pointeur dans `Dream-Toolkit/README.md`.
- [x] Description de `SPEC.md` (*Project overview*) : élargie (inspection, harmonisation, prétraitement,
      extraction de mesures, EDF + Curry expérimental) + un paragraphe **Name** (ancien nom, environnement).
- [ ] **Toi, à la main** : les tutoriels `.docx`, en particulier `tutorials/1_install_python.docx` (4 mentions
      de `inspect_edf`, la commande périmée `voila inspect_edf_voila.ipynb` → `voila tools/1_inspect_edf_voila.ipynb`,
      **les captures d'écran**, et l'installation légère / complète une fois la Phase 6 tranchée).
- [x] **Toi, plus tard** : « SomnoLens » sur le poster (tu le modifies directement dans PowerPoint). ==> déjà fait

Ne pas toucher : `tools/old/`, `tools/jupyter_notebook/` (archives gelées), `.ipynb_checkpoints/`, les sorties
de données (`tools/test_data/summary_inspection/`, `tools/ICEBERG_snippet/`, rapports HTML), qui contiennent
d'anciens chemins absolus (voir Phase 5).

### Phase 3 : mettre l'arborescence de SPEC.md à jour (indépendant du renommage, fait le 2026-10-29)
- [x] **Listé mais absent** : `tools/preprocessing_phase1_example_scripts/` → supprimer la ligne.
- [x] **Versionnés mais absents de l'arborescence** : `README.md` (ex-`README.docx`), `pytest.ini`,
      `tutorials/` (4 guides Word : installation, montage Profusion, export EDF), `tests/outside_chain.py`,
      `tests/refactor_check.py`, + `tests/score_edf_equivalence.py`.
- [x] Vérifier que les noms de dossiers de sortie cités dans l'arbre et dans *How to run* suivent les
      renommages récents (`preproc_epo/`, `reports_preproc/`, `reports_inspection/`) : à jour ; les anciens noms
      restants (`raw_epo`, `clean_epo_*`) sont des notes de migration, gardées.
- [ ] À ta décision : l'arborescence cite deux fichiers **non versionnés** (`tools/sleep_metrics_formulas.txt`,
      les définitions des métriques de l'outil 5, et `tools/images/_make_unicorn_gif.py`). Les versionner, ou
      retirer la mention ?

### Phase 4 : outils annexes
- [x] Mes mémoires Claude (6 fichiers citaient `Inspect_EDF`/`inspect_edf`) : mises à jour le 2026-10-29. Elles sont
      rangées sous `Dream-Toolkit`, donc le renommage ne les perd pas.
- [ ] **Conversations Claude Code** : elles sont rangées par dossier, dans
      `~/.claude/projects/c--Users-…-Dream-Toolkit-python-Inspect-EDF/` (14 conversations aujourd'hui).
      Pour les retrouver dans `/resume` depuis le nouveau dossier, on les **copie** (sans supprimer l'original)
      dans le dossier équivalent `…-python-SomnoLens/`, que Claude Code crée à la première session lancée
      depuis `SomnoLens`. On teste d'abord avec une seule conversation, puis on copie le reste si elle apparaît.
- [ ] `.claude/settings.local.json` ne contient aucun chemin vers `Inspect_EDF` : rien à faire.

### Phase 5 : vérifications
- [ ] `grep -rn "Inspect_EDF\|inspect_edf"` sur les fichiers actifs : il ne doit rester que les noms de
      fichiers de l'outil 1 et les archives.
- [ ] Tests **quick** (`-m quick`) depuis le nouveau dossier avec le nouvel environnement : le renommage ne
      touche aucun calcul, et `nbdriver` calcule la racine du dépôt à partir de son propre emplacement. Le cache
      de la chaîne (`%TEMP%\dtk_tests`) reste valable (empreintes = contenu des fichiers, pas leurs chemins).
- [ ] Lancer une app Voila à la main (ex. `voila tools/1_inspect_edf_voila.ipynb`).
- [ ] Effet attendu, sans gravité : l'outil 1 compare les fichiers sur leur chemin absolu (`path` de
      `FULL_summary_table_edf.tsv`). Les jeux de données **rangés dans le dépôt** (`tools/test_data`,
      `tools/ICEBERG_snippet`) seront donc ré-inspectés une fois. Les études rangées ailleurs ne changent pas.
- [ ] Proposer le message de commit (tu commites toi-même), puis supprimer ce plan dans un commit à part.

### Phase 6 : réflexions à mener (pas forcément dans ce chantier)
**6a. Prendre une option sur le nom** (possible dès maintenant, indépendamment du renommage) :
- [x] **GitHub** : dépôt `SomnoLens` créé le 2026-10-16 avec un README descriptif et la licence **BSD-3**
      (choix provisoire).
- [ ] ⚠ **Licence à confirmer avec le service juridique / valorisation de l'ICM** : en tant que salarié, le code
      appartient en principe à l'employeur. Demander qui est le titulaire du copyright (ligne `Copyright (c) …`
      du fichier `LICENSE`) et si une licence est imposée (les organismes français recommandent souvent
      **CeCILL-B**, l'équivalent BSD conforme au droit français).
- [ ] **PyPI** : le nom `somnolens` est libre, mais PyPI interdit de réserver un nom avec un paquet vide
      (règle PEP 541 : un nom inutilisé peut être réattribué). On le prend en publiant une **première version
      réelle**, même minimale (par exemple l'inspecteur d'en-têtes EDF), le jour où la distribution est décidée (6b).
- [ ] Vérification de marque (TMview pour l'UE, INPI pour la France) avant une publication officielle.
- [ ] (Optionnel) un nom de domaine : somnolens.com est pris ; .org / .io / .fr à vérifier chez un registraire.

**6b. Comment distribuer les outils** (aujourd'hui, ils vivent dans le dépôt global de l'équipe, ce qui ne
rend pas la distribution très classique ni intuitive). Pistes à comparer, rien n'est décidé :
- [ ] **Dépôt GitHub dédié `SomnoLens`**, extrait du dépôt d'équipe en **gardant l'historique** (`git subtree
      split`) ; le dépôt d'équipe le référence (sous-module, ou simple lien).
- [ ] **Paquet installable** (`pip install somnolens`) avec une commande qui ouvre un menu des outils Voila ;
      les données de test et les archives restent hors du paquet.
- [ ] **Installation Windows « en un clic »** pour les non-programmeurs : un script `.bat` qui crée
      l'environnement et lance l'outil 1, en complément des tutoriels.

**6c. Installation légère / complète** (à trancher avant de réécrire le tutoriel d'installation) :
- [ ] **Légère** = tous les outils sauf l'autoscoring : l'environnement actuel **moins** `pytorch-cpu`, `gssc` et
      les paquets de SleepFM (`einops`, `loguru`, `pyyaml` ; `h5py` reste, `mne` l'apporte). `yasa` reste aussi
      (léger, sans torch). À trancher : YASA seul dans la légère suffit-il pour proposer l'outil 3bis sans torch ?
- [ ] **Complète** = légère **+** `pytorch-cpu` + `gssc` + les paquets de SleepFM (outil 3bis). Plus besoin de
      `python/Score_EDF/` : son calcul est porté dans 3bis (2026-10-27). Le modèle SleepFM (~29 Mo) n'est pas dans
      l'environnement : il se télécharge au premier usage. Le poids vient surtout de `pytorch-cpu`.
- [x] ~~Conflit de versions~~ **résolu pour l'outil 3bis** (testé le 2026-10-27 et le 2026-10-29, nuit 100) : les
      trois modèles tournent dans l'environnement SomnoLens (python 3.12, numpy 2.2, torch 2.8, yasa 0.6.5) avec
      les mêmes résultats que dans l'environnement `autoscore` de Score_EDF (python 3.10, numpy 1.26, torch 2.5.1,
      yasa 0.7.0). Torch < 2.6 et yasa ≥ 0.7 ne sont plus nécessaires.
      | Modèle | SomnoLens vs `autoscore` |
      |---|---|
      | SleepFM | identique : 5 652 pas de 5 s, écart max des probabilités 2e-6, aucun stade différent |
      | GSSC | identique : 941 époques, aucun stade différent (probabilités égales aux 4 décimales gardées) |
      | YASA | quasi identique : 2 époques sur 941 diffèrent sur C3-M2 (κ 0,51 des deux côtés) ; 190 sur la mauvaise dérivation C4-A1 de cette nuit (modèle incertain : un petit écart de feature change le stade) |
      Seul le code d'origine de Score_EDF n'a pas tourné dans l'environnement SomnoLens. Il n'en a plus besoin
      (option A du plan `plan_score_edf_standalone.md` : le reconstruire sur les bibliothèques de 3bis).
- [ ] ⚠ **YASA dépend de la version d'`antropy`** (cause de l'écart ci-dessus) : antropy 0.2 a corrigé une erreur
      d'un pas dans la dimension fractale de Higuchi (`range(1, n_max)` → `range(1, n_max + 1)`). Les classifieurs de
      YASA (fichiers `_0.5.0`, 2021) ont été entraînés avec l'ancien calcul, que garde l'antropy 0.1.9 de SomnoLens.
      Épingler `antropy<0.2` dans l'installation (YASA reste alors tel qu'il a été entraîné, et reproductible).
- [ ] ⚠ **YASA 0.8 cassera l'outil 3bis** : `predict_proba()` (utilisé par `score_night_yasa`) est déprécié en 0.7
      et retiré en 0.8. Garder `yasa=0.6.*`, ou passer à `predict().proba` avant de monter de version.
- [ ] **Rafraîchir toutes les versions avec ce chantier** (décision du 2026-10-29) : versions fixées, mises à jour
      délibérément (une à deux fois par an, quelques semaines après une sortie), chaque mise à jour validée par la
      suite de tests complète, une étude traitée avec une seule version. Pour ce premier rafraîchissement, toutes
      les versions passent à leur dernière version stable, YASA 0.8 compris : `predict().proba`, pandas ≥ 2.3,
      import de `lspopt` en direct, `antropy<0.2`, `antropy_version` dans les paramètres. Détail : SPEC
      *Planned modules → Lighter conda environments*, point *Version refresh*.
- [ ] Le message de `autoscoring_lib.py` quand GSSC manque (`GSSC_MSG`), et ceux de YASA (`YASA_MSG`) et de
      SleepFM (`SLEEPFM_DEPS_MSG`), devront pointer vers l'installation complète.
- [ ] ⚠ **Licences à vérifier avant de distribuer l'installation complète** (vérifié sur GitHub le 2026-10-16) :
      | Dépendance | Licence |
      |---|---|
      | MNE-Python, YASA, Voila | BSD-3 (compatibles avec une licence BSD-3) |
      | specparam, edfio | Apache-2.0 (compatibles) |
      | **GSSC** (outil 3bis) | **AGPL-3.0** : copyleft strict, y compris pour un usage à travers un réseau |
      - Le code SomnoLens peut rester en BSD-3 tant que GSSC est une dépendance **installée séparément**, mais
        quiconque distribue SomnoLens **avec** GSSC doit respecter l'AGPL pour l'ensemble.
      - `tools/autoscoring_lib.py` (`gssc_combination_logits`) **reproduit la boucle de scoring de GSSC** : si c'est
        une copie de leur code, ce fichier relève lui-même de l'AGPL. À isoler (module optionnel) ou à
        licencier à part.
      - **SleepFM** (outil 3bis) : code et poids sous **CC BY-NC 4.0** (usage **non commercial**, vu sur GitHub lors
        du portage). SomnoLens ne les redistribue pas : ils se téléchargent depuis le dépôt officiel au premier usage.
        À signaler dans la documentation de l'outil 3bis.
      - Vérifier aussi les licences des autres paquets de l'installation complète (pytorch, `einops`, `loguru`,
        `pyyaml`), plus ceux de `Score_EDF` si sa version autonome est distribuée.
      - Faire relire ce point par le service juridique de l'ICM (en même temps que 6a).

## Après validation
Supprimer ce fichier une fois le renommage fait et vérifié (dans un commit à part).
