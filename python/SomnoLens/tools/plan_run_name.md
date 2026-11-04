# Plan : plusieurs runs de l'outil 7, avec un nom de run

Statut : validé le 2026-11-04 (option A + comparaison des contenus), en cours. Remplace le TODO de SPEC *Renaming the outputs, several channel configurations of tool 2*.

## 1. Objectif

Aujourd'hui, l'outil 7 écrit toujours dans `derivatives/preproc_epo/` et `reports_preproc/`. Pour garder un run et en
lancer un autre (d'autres filtres, une autre configuration de canaux), il faut renommer les dossiers à la main, et les
sorties des outils 8, 8bis et 9 restent communes aux deux runs.

Après cette tâche :
- l'outil 7 voit les runs déjà présents et propose d'en créer une nouvelle version (`_v2`, `_v3`…) ou un nom libre ;
- les outils 8, 8bis et 9 suivent : chaque run a ses décisions et ses features, sans rien renommer à la main ;
- une étude avec un seul run ne change pas : mêmes noms de dossiers, sorties identiques au bit près.

## 2. Ce que l'utilisateur voit dans l'outil 7

Section 3 (*Participants*), sous le sélecteur, à côté de la case *Skip already processed* : c'est là que le nombre de
participants à traiter se décide, et aucun nouveau Scan n'est nécessaire.

Premier run de l'étude : une ligne seule, pas de case.

```
Run: preproc_epo (first run)
     writes derivatives/preproc_epo/ and reports_preproc/
```

Un run existe déjà :

```
Run: preproc_epo (11 participants)                 (un menu s'il y en a plusieurs, le plus récent pré-choisi)
☑ Skip already processed participants
☐ New version (another run)
     To run the analyses with other parameters and keep the outputs of preproc_epo.
```

Case cochée, le champ apparaît :

```
☑ New version (another run)
     To run the analyses with other parameters and keep the outputs of preproc_epo.
     Suffix: [_v2]   writes derivatives/preproc_epo_v2/ and reports_preproc_v2/
```

- **Retraiter un run existant** : décocher *Skip*, comme aujourd'hui. Pas de choix *Overwrite* en plus.
- **Le suffixe** : le champ propose le premier `_vN` libre, on peut écrire autre chose (`_nonotch`). Lettres, chiffres,
  `-` et `_` seulement. Un suffixe invalide ou un run qui existe déjà : un message ⚠ et le Run refuse.
- **Un suffixe encore présent dans le registre** (run dont les dossiers ont été supprimés à la main) n'est pas reproposé :
  le nouveau run hériterait de ses exclusions.
- **Le comptage** *N already processed* et le Skip portent sur le run choisi. Changer de run ou cocher la case les
  recalcule tout de suite.
- **Après le Run** : le run créé devient le run choisi, la case se décoche, le champ propose le suffixe suivant.
- **Les exclusions** : la case cochée nomme les exclusions manuelles du run choisi, que la nouvelle version ne reprend pas.
- Avec un dossier de sortie (Advanced), les runs sont cherchés dans ce dossier.

## 3. Ce qui change dans les sorties

Le suffixe du run se lit dans le nom du dossier d'époques (`preproc_epo_v2` → `_v2`) et se répète partout :

| Outil | Run par défaut (inchangé) | Run `_v2` |
|---|---|---|
| 7 | `derivatives/preproc_epo/`, `reports_preproc/` | `derivatives/preproc_epo_v2/`, `reports_preproc_v2/` |
| 8bis | `derivatives/rejection_auto/`, `reports_rejection_auto/` | `derivatives/rejection_auto_v2/`, `reports_rejection_auto_v2/` |
| 8 | `derivatives/rejection_manual/`, `reports_rejection_manual/` | `derivatives/rejection_manual_v2/`, `reports_rejection_manual_v2/` |
| 9 | `derivatives/features_spectral/`, `reports_features_spectral/` | `derivatives/features_spectral_v2/`, `reports_features_spectral_v2/` |

- Outils 8, 8bis, 9 : le menu des runs existe déjà (phase 3b). Il passe du premier par ordre alphabétique au **plus
  récent** pré-choisi, la ligne dit pourquoi.
- Outil 9 : le menu des décisions ne montre que celles **du run choisi** (`rejection_auto_v2` pour `preproc_epo_v2`).
  « Choose another… » reste possible.
- Les outils 5, 5bis, 6 et 0 ne changent pas (l'outil 0 lit le run par défaut).
- Une autre configuration de canaux : on choisit le fichier avec « Choose another… » puis *New version*. C'est la réponse
  à la question ouverte du TODO (garder séparées les sorties de plusieurs configurations).

## 4. Décisions

1. **Numérotation (décidé).** Le premier run garde `preproc_epo` sans suffixe, le suivant est `_v2`.
2. **Quand un run existe (décidé).** La case *New version* est décochée : on continue le run existant, c'est le
   comportement actuel. L'outil ne sait pas encore si les réglages ont changé. Ce sera possible avec le TODO *Skip only
   what was made with the current settings* : il pourrait alors le signaler.
3. **Registre d'exclusions (décidé).** Un registre commun mélangerait les runs : l'outil 8bis sur `_v2` remplace la
   ligne `8bis_auto` du participant pour tous les runs, et revenir ensuite à l'ancien run lui applique les décisions du
   `_v2`. Les runs se font plutôt l'un après l'autre, puis se comparent. Proposition :
   - **Un seul fichier, une règle** : ce qui est décidé avant le prétraitement (5bis) vaut pour l'étude, ce qui est décidé
     à partir de l'outil 7 appartient au run (décidé pour les exclusions manuelles). Une inclusion forcée de 5bis vaut donc
     pour tous les runs.
   - **Le run est dans la source** : `7_manual_v2`, `8_manual_v2`, `8bis_auto_v2`. Le run par défaut garde `7_manual`,
     `8_manual`, `8bis_auto` : pas de nouvelle colonne, pas de migration, rien ne change pour une étude à un run.
   - **Lecture** : un outil sur `_v2` lit les lignes de 5bis et celles de `_v2`, et ignore les autres. Les outils sans
     choix de run (0, 5bis, 6) lisent les lignes de 5bis et celles du run par défaut.
   - **Pour comparer** : un seul fichier montre côte à côte ce que chaque run a décidé pour un participant, et les tables
     de chaque run portent ses propres `participant_excluded`.
   - **Plus de fausse alerte** : l'encadré orange *registry changed* ne regarde plus la date du fichier. Il compare ce que
     disent les tables du run (`participant_excluded`, `participant_exclude_reason`) à ce que dit le registre pour ce run.
     Une décision sur `_v2` ne le fait donc plus apparaître sur l'autre run. Une décision de 5bis le fait apparaître sur
     tous, c'est juste. Une décision annulée entre-temps ne le fait plus apparaître non plus. Vaut pour les outils 6, 7,
     8bis et 9.
   - **Limite** : deux pages lancées au même moment écrivent le même fichier, une ligne peut se perdre si deux écritures
     tombent dans la même fraction de seconde. Risque déjà présent aujourd'hui, peu probable vu l'usage.
   - **Écartés** : un fichier par run (`participant_exclusions_v2.tsv`) : les lignes du run par défaut restent dans le
     fichier principal avec celles de 5bis, sauf à migrer les études existantes. Il faut ouvrir plusieurs fichiers pour
     comparer, et chaque outil en lit deux. Son seul gain est la limite ci-dessus. Une copie du registre à la création du
     run : une décision 5bis prise ensuite n'atteindrait pas les copies. Accepter le mélange avec un avertissement : ni
     comparaison fiable, ni retour à un ancien run.
4. **Dossiers déjà renommés à la main (décidé).** `preproc_epo_v2` avec ses décisions dans `rejection_auto` : le run est
   reconnu, ses nouvelles décisions iront dans `rejection_auto_v2`, les anciennes restent accessibles par « Choose
   another… ». Rien n'est migré. Ses lignes déjà écrites au registre restent celles du run par défaut.

## 5. Ce qui change dans le code

- `study_files_lib` : `run_suffix(epoch_folder)`, `next_run_suffix(root)`, `valid_run_name(text)`, `epoch_folders` trié
  du plus récent au plus ancien, `decision_folders` limité au run.
- `participant_selection_lib` : `effective_status(reg, run='')` garde les lignes de 5bis et celles du run (les outils 0,
  5bis, 6 ne changent donc pas pour la lecture), `write_registry_rows` accepte une source suffixée,
  `tables_out_of_date(status, *tables)` remplace `registry_newer_than` (outils 6, 7, 8bis, 9). Lectures : outils 7
  (1 endroit), 8, 8bis, 9 (2 chacun). Écritures : outils 7 et 8 (3 chacun), 8bis (1).
- Outil 7, section 3 : la ligne du run (`UI.choice_menu` s'il y en a plusieurs), la case et son champ, une fonction
  `run_suffix()` lue aux endroits qui écrivent `preproc_epo` et `reports_preproc` (7 endroits). Le calcul des participants
  déjà traités sort du Scan dans une fonction, relancée quand le run change. La migration de `raw_epo` reste réservée au
  run par défaut.
- Outil 8bis : `DECISION_DIRNAME` et `reports_rejection_auto` (5 endroits). Outil 8 : `rejection_manual` et
  `reports_rejection_manual` (4 endroits). Outil 9 : `DATA_DIRNAME` et `REPORTS_DIRNAME` (2 endroits).
- Jumeaux Curry 7, 8, 8bis : générateurs relancés, le code est indépendant du format.
- SPEC (*Study folder layout*, §7, §8, §8bis, §9, le TODO retiré), CLAUDE.md (les rappels sur les sorties des outils 7,
  8, 8bis, 9), README si besoin, TESTS.md.

## 6. Tests

- `test_study_files_lib` (quick) : le suffixe d'un dossier, le premier `_vN` libre, les noms refusés, le tri, le filtre
  des décisions.
- `test_participant_selection_lib` (quick) : les lignes de deux runs, le statut lu par chacun, la forcée de 5bis qui vaut
  pour les deux, une écriture sur `_v2` qui laisse intactes celles du run par défaut. Les tables périmées : une décision
  du run oui, une décision d'un autre run non, une décision annulée non.
- `test_inputs_found` (standard) : la section 3 de l'outil 7 sur des fichiers vides (premier run, un run, plusieurs runs,
  suffixe proposé, libre, refusé, déjà pris, comptage recalculé).
- `test_variants` (full, données synthétiques) : un run `_v2` de l'outil 7 sur un participant, puis 8bis et 9 dessus. Le
  run par défaut n'est pas touché, et les sorties du `_v2` sont dans leurs dossiers. Une exclusion 8bis sur `_v2` n'apparaît
  pas dans les tables reconstruites du run par défaut.
- La chaîne reste identique à ses références : c'est le test que le run par défaut ne change pas.
- Niveau proposé : standard pendant le travail, full à la fin.

## 7. Hors périmètre

- Détecter que les réglages ont changé par rapport au run existant (TODO *Skip only what was made with the current
  settings*).
- Un run pour les outils 5 et 6 : pas de demande, et leurs sorties ne dépendent pas du prétraitement.
- Comparer deux runs entre eux.
