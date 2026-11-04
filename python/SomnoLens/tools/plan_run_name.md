# Plan : plusieurs runs de l'outil 7, avec un nom de run

Statut : points 1, 2 et 4 décidés, point 3 à décider (§4). Remplace le TODO de SPEC *Renaming the outputs, several channel configurations of tool 2*.

## 1. Objectif

Aujourd'hui, l'outil 7 écrit toujours dans `derivatives/preproc_epo/` et `reports_preproc/`. Pour garder un run et en
lancer un autre (d'autres filtres, une autre configuration de canaux), il faut renommer les dossiers à la main, et les
sorties des outils 8, 8bis et 9 restent communes aux deux runs.

Après cette tâche :
- l'outil 7 voit les runs déjà présents et propose d'en créer une nouvelle version (`_v2`, `_v3`…) ou un nom libre ;
- les outils 8, 8bis et 9 suivent : chaque run a ses décisions et ses features, sans rien renommer à la main ;
- une étude avec un seul run ne change pas : mêmes noms de dossiers, sorties identiques au bit près.

## 2. Ce que l'utilisateur voit dans l'outil 7

Section 1, une ligne **Run** sous la liste des entrées (visible, pas dans Advanced : c'est au moment où un run existe
que la question se pose).

Premier run de l'étude :

```
Run: [ New run: preproc_epo            ▾]      (autre choix du menu : Other name…)
     writes derivatives/preproc_epo/ and reports_preproc/
```

Un run existe déjà :

```
Run: [ Continue preproc_epo (11 participants)  ▾]
     Autres choix du menu :  Overwrite preproc_epo (11 participants)
                             New version: preproc_epo_v2
                             Other name…
     ℹ️ A run already exists. Continue: adds the new participants, skips the processed ones. Overwrite: reprocesses them
        in this folder. To keep it and try other settings, choose New version.
```

- Plusieurs runs : une ligne *Continue* par run, avec le nombre de participants, **le plus récent pré-choisi**.
- *Continue* et *Overwrite* ne font que régler la case *Skip already processed* (cochée, décochée), qui reste ce que le
  Run lit. Changer la case à la main change le menu, et inversement. *Overwrite* affiche un ⚠ : les décisions des outils
  8 / 8bis et les features de l'outil 9 de ce run viennent des anciennes époques, il faut les relancer, *Skip* décoché.
  Rien n'est supprimé d'avance : seuls les participants retraités sont remplacés (avec un subset, ceux du subset).
- *New version* propose le premier `_vN` libre (dans `derivatives/` et à la racine pour les rapports).
- *Other name…* fait apparaître un champ : `nonotch` donne `preproc_epo_nonotch` et `reports_preproc_nonotch`. Lettres,
  chiffres, `-` et `_` seulement, sinon un message ⚠ et le Run refuse.
- Le comptage *N / M already have a report* et le Skip portent sur le run choisi. Changer de run après un Scan demande un
  nouveau Scan (message dans la section 3), comme un autre choix en amont.
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

1. **Numérotation (décidé).** Le premier run garde `preproc_epo` sans suffixe, le suivant est `_v2`. Quand un run existe,
   l'outil propose de le continuer, de l'écraser ou de faire un `_v2` (§2).
2. **Choix par défaut quand un run existe (décidé).** *Continue* : ajouter des participants à une étude est le cas le plus
   courant, et c'est le comportement actuel. L'outil ne sait pas encore si les réglages ont changé. Ce sera possible avec
   le TODO *Skip only what was made with the current settings* : il pourrait alors suggérer *New version* dans ce cas.
3. **Registre d'exclusions (à décider, ma proposition).** Un registre commun mélangerait les runs : l'outil 8bis sur `_v2`
   remplace la ligne `8bis_auto` du participant pour tous les runs, et revenir ensuite à l'ancien run lui applique les
   décisions du `_v2`. Proposition :
   - **Un seul fichier, une règle** : ce qui est décidé avant le prétraitement (5bis) vaut pour l'étude, ce qui est décidé
     à partir de l'outil 7 appartient au run. Une inclusion forcée de 5bis vaut donc pour tous les runs.
   - **Le run est dans la source** : `7_manual_v2`, `8_manual_v2`, `8bis_auto_v2`. Le run par défaut garde `7_manual`,
     `8_manual`, `8bis_auto` : pas de nouvelle colonne, pas de migration, rien ne change pour une étude à un run.
   - **Lecture** : un outil sur `_v2` lit les lignes de 5bis et celles de `_v2`, et ignore les autres. Les outils sans
     choix de run (0, 5bis, 6) lisent les lignes de 5bis et celles du run par défaut.
   - **Une exclusion à la main ne suit pas d'un run à l'autre** : pour un enregistrement inutilisable dans tous les runs,
     c'est la section 3 de 5bis. L'outil 7 nomme, à la création d'une version, les exclusions manuelles des autres runs
     qu'elle ne reprend pas.
   - **Limite 1** : l'encadré orange *registry changed* regarde la date du fichier. Une décision sur `_v2` le fait donc
     apparaître aussi sur l'autre run. Le rebuild qu'il propose n'y change rien : fausse alerte, sans danger.
   - **Limite 2** : deux pages lancées au même moment écrivent le même fichier. Une ligne peut se perdre si deux écritures
     tombent dans la même fraction de seconde. Ce risque existe déjà aujourd'hui (8 et 8bis côte à côte), il ne change pas.
   - **Écartés** : un fichier par run (règle la limite 2, mais chaque outil lit deux fichiers pour le même résultat), une
     copie du registre à la création du run (une décision 5bis prise ensuite n'atteindrait pas les copies), accepter le
     mélange avec un avertissement (il ne permet ni le parallèle, ni de revenir à un ancien run).
4. **Dossiers déjà renommés à la main (décidé).** `preproc_epo_v2` avec ses décisions dans `rejection_auto` : le run est
   reconnu, ses nouvelles décisions iront dans `rejection_auto_v2`, les anciennes restent accessibles par « Choose
   another… ». Rien n'est migré. Ses lignes déjà écrites au registre restent celles du run par défaut.

## 5. Ce qui change dans le code

- `study_files_lib` : `run_suffix(epoch_folder)`, `next_run_suffix(root)`, `valid_run_name(text)`, `epoch_folders` trié
  du plus récent au plus ancien, `decision_folders` limité au run.
- `participant_selection_lib` (si le point 3 est retenu) : `effective_status(reg, run='')` garde les lignes de 5bis et
  celles du run (les outils 0, 5bis, 6 ne changent donc pas), `write_registry_rows` accepte une source suffixée. Lectures :
  outils 7 (1 endroit), 8, 8bis, 9 (2 chacun). Écritures : outils 7 et 8 (3 chacun), 8bis (1).
- Outil 7 : le menu du run (sur le modèle de `UI.suffix_menu` : un choix *Other name…* qui montre le champ), lié à la case
  *Skip*, une fonction
  `run_suffix()` lue aux endroits qui écrivent `preproc_epo` et `reports_preproc` (7 endroits). La migration de `raw_epo`
  reste réservée au run par défaut.
- Outil 8bis : `DECISION_DIRNAME` et `reports_rejection_auto` (5 endroits). Outil 8 : `rejection_manual` et
  `reports_rejection_manual` (4 endroits). Outil 9 : `DATA_DIRNAME` et `REPORTS_DIRNAME` (2 endroits).
- Jumeaux Curry 7, 8, 8bis : générateurs relancés, le code est indépendant du format.
- SPEC (*Study folder layout*, §7, §8, §8bis, §9, le TODO retiré), CLAUDE.md (les rappels sur les sorties des outils 7,
  8, 8bis, 9), README si besoin, TESTS.md.

## 6. Tests

- `test_study_files_lib` (quick) : le suffixe d'un dossier, le premier `_vN` libre, les noms refusés, le tri, le filtre
  des décisions.
- `test_participant_selection_lib` (quick, si le point 3 est retenu) : les lignes de deux runs, le statut lu par chacun, la
  forcée de 5bis qui vaut pour les deux, une écriture sur `_v2` qui laisse intactes celles du run par défaut.
- `test_inputs_found` (standard) : le menu de l'outil 7 sur des fichiers vides (premier run, un run, plusieurs runs,
  *Overwrite* et la case *Skip*, nom libre, nom refusé).
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
