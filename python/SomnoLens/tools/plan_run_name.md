# Plan : plusieurs runs de l'outil 7, avec un nom de run

Statut : à valider. Remplace le TODO de SPEC *Renaming the outputs, several channel configurations of tool 2*.

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
     Autres choix du menu :  New version: preproc_epo_v2
                             Other name…
     ℹ️ A run already exists. Continue: adds the new participants, skips the processed ones. To keep it and try other
        settings, choose New version.
```

- Plusieurs runs : une ligne *Continue* par run, avec le nombre de participants, **le plus récent pré-choisi**.
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

## 4. Points à décider (mes recommandations)

1. **Numérotation.** Tu as évoqué `_v1`, `_v2`. Je recommande : le premier run garde `preproc_epo` sans suffixe, le
   suivant est `_v2`. Sinon toutes les études existantes changeraient de nom, et un run seul porterait un `_v1` inutile.
2. **Choix par défaut quand un run existe.** Je recommande *Continue* : ajouter des participants à une étude est le cas
   le plus courant, et c'est le comportement actuel. L'outil ne sait pas encore si les réglages ont changé. Ce sera
   possible avec le TODO *Skip only what was made with the current settings* : il pourrait alors suggérer *New version*
   dans ce cas.
3. **Registre d'exclusions.** Il reste **unique pour l'étude**, pas un par run. Un participant exclu à la main l'est pour
   tous les runs. Limite : les exclusions automatiques de 8bis (`8bis_auto`) sont remplacées à chaque lancement, donc
   celles du dernier run lancé s'appliquent à tous. Je recommande d'accepter cette limite et de l'afficher dans 8bis
   quand l'étude a plusieurs runs. Un registre par run toucherait aussi les outils 5bis, 6, 7, 8 et 9.
4. **Dossiers déjà renommés à la main** (`preproc_epo_v2` avec ses décisions dans `rejection_auto`). Le run est reconnu.
   Ses nouvelles décisions iront dans `rejection_auto_v2`. Les anciennes restent accessibles par « Choose another… ». Je
   recommande de ne rien migrer.

## 5. Ce qui change dans le code

- `study_files_lib` : `run_suffix(epoch_folder)`, `next_run_suffix(root)`, `valid_run_name(text)`, `epoch_folders` trié
  du plus récent au plus ancien, `decision_folders` limité au run.
- Outil 7 : le menu du run (sur le modèle de `UI.suffix_menu` : un choix *Other name…* qui montre le champ), une fonction
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
- `test_inputs_found` (standard) : le menu de l'outil 7 sur des fichiers vides (premier run, un run, plusieurs runs, nom
  libre, nom refusé).
- `test_variants` (full, données synthétiques) : un run `_v2` de l'outil 7 sur un participant, puis 8bis et 9 dessus. Le
  run par défaut n'est pas touché, et les sorties du `_v2` sont dans leurs dossiers.
- La chaîne reste identique à ses références : c'est le test que le run par défaut ne change pas.
- Niveau proposé : standard pendant le travail, full à la fin.

## 7. Hors périmètre

- Détecter que les réglages ont changé par rapport au run existant (TODO *Skip only what was made with the current
  settings*).
- Un run pour les outils 5 et 6 : pas de demande, et leurs sorties ne dépendent pas du prétraitement.
- Comparer deux runs entre eux.
