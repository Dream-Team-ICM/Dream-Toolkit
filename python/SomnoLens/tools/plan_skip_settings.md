# Plan : Skip ne saute que ce qui a été fait avec les réglages courants (outils 6, 7, 8bis, 9 et aval)

**Statut** : à relire, version 2 avec tes remarques du 2026-11-05. Rien n'est implémenté avant ta confirmation.
**Contexte** : SPEC, *Planned modules*, point 1 de l'ordre des TODO. L'outil 8bis compare déjà ses réglages depuis le
2026-11-02 (`autoreject_lib.settings_changed`). Les outils 6, 7 et 9 ont le même piège : de nouveaux réglages avec *Skip*
coché ne s'appliquent pas aux participants déjà traités.

## Décisions prises (2026-11-05)

1. **Une différence arrête Run, qui demande quoi faire.** Le contrôle a lieu juste après le clic sur Run, **avant le premier
   participant** : aucune analyse ne démarre et aucune sortie n'est écrite tant que tu n'as pas répondu. Il lit seulement
   ce que les outils ont déjà écrit (sidecars, tables par participant), jamais le signal.
2. **Outil 6, seuils** : les flags sont recalculés depuis les métriques des tables. Seuls les participants dont un flag
   change sont concernés.
3. **Outil 7, canaux** : pour un participant déjà traité dans ce run, l'éditeur montre les canaux enregistrés dans le
   sidecar, puis on compare au Run.
4. **Aval** : traité en phase 4 (empreinte des réglages de l'outil 7 dans les outils 8, 8bis et 9).
5. **L'outil 8bis reçoit aussi l'encadré**, pour que les quatre outils se comportent pareil. Il a le même piège à la
   réouverture : ses seuils reviennent aux valeurs par défaut (phase 4).
6. **La reprise des réglages enregistrés fait partie de ce plan** (phase 5).

**Choix que j'ai faits sans te les demander** (à corriger ici si besoin) :

- *Skip* décoché et *Re-aggregate only* ne changent pas : ce sont déjà des demandes explicites.
- Pas d'enregistrement à comparer (sortie ancienne, sidecar absent) : le participant reste sauté, sans encadré, et le log
  donne leur nombre en ℹ️. Un ancien run n'est donc jamais traité comme différent sans preuve.
- Enregistrement illisible : il apparaît dans l'encadré (`unreadable record`), comme le fait déjà l'outil 8bis. Le contrôle
  ne fait jamais échouer le run : une erreur sur un participant donne un ⚠️ et le contrôle continue.

## Constats à la lecture du code

- **Le `⚠ made with…` de l'outil 6 ne sort presque jamais.** `selected_ids()` retire les participants déjà traités avant
  la boucle, donc dans une session ouverte sur une étude existante la boucle ne les voit pas. Le test
  `test_tool6_resampling_and_highpass` passe car ses deux runs ont lieu dans la même session, où la liste n'est pas
  rafraîchie. Le `⤼` de la boucle de l'outil 7 a le même défaut.
- **L'alerte « mixed-method » de l'outil 7 lit tous les runs** (`derivatives/` en entier) : un run `_v2` peut déclencher
  une fausse alerte. Elle sera limitée au run choisi.
- **Le rééchantillonnage demandé après une table sans rééchantillonnage n'est pas détecté** par
  `signal_preparation_mismatch` (la fréquence native manque). L'outil 6 la connaît pourtant par son scan d'acquisition.
- **Les seuils de l'outil 6 ne sont enregistrés nulle part**, sauf en partie dans le texte de `suspect_reason`. Il faut les
  enregistrer pour pouvoir les reprendre (phase 1).

## Mécanisme commun (`participant_selection_lib`)

Au clic sur Run, avec *Skip* coché, avant toute analyse :

1. Les candidats sont les participants choisis (`chosen_ids()`) dont les sorties sont **sur le disque**, pas la liste
   `done` du sélecteur, qui peut dater du dernier Scan.
2. Chaque outil compare ses réglages à ce que le participant a enregistré. Sa fonction renvoie une liste
   `(réglage, valeur enregistrée, valeur courante)`, vide si tout est identique.
3. **Aucune différence** : le run se déroule comme aujourd'hui.
4. **Au moins une différence** : le run s'arrête et affiche, dans sa zone de log :

```
⚠️ 12 of the 30 processed participants were made with other settings
   high-pass: off → 0.5 Hz      12 participants (165, 176, …)
   notch: 50 Hz → off            3 participants (180, 181, …)
   Keeping them mixes settings in the database tables.

   [Reprocess these 12 + run the 3 new ones]   [Keep them, run the 3 new ones]
   [Use the recorded settings]
```

- Les réglages sont regroupés par valeurs identiques, avec 10 noms au plus puis `(+k)`, comme les menus de suffixes.
- Sans nouveau participant, le second bouton devient *Keep them, only rebuild the tables* : Run avec Skip coché reste la
  façon de reporter une décision du registre dans les tables.
- La comparaison est refaite à chaque clic sur Run. Le choix ne vaut que pour ce run.
- Boutons : `participant_selector.btn_redo_other` (`caution`), `btn_keep_other` (`run`), et `btn_use_recorded` (`helper`,
  phase 5), pilotables par les tests. Le premier Run se termine simplement après l'affichage.
- Le log final nomme les participants refaits et leurs réglages, ou ceux gardés.

## Phase 1 : encadré commun et outil 6 (+ jumeau Curry)

Nouvelle fonction `quality_lib.settings_differences(stored_rows, …)`, copiée dans le jumeau par son générateur :

| Réglage | Comparé à | Règle |
|---|---|---|
| Préparation du signal | colonne `signal_preparation` | Logique de `signal_preparation_mismatch`, avec en plus un paramètre `native_sf` : un rééchantillonnage demandé compte si la cible est sous la fréquence native. Cellule vide : rien à comparer. |
| Seuils de flag | `suspect_reason` du `_quality_metrics.tsv` | `flag_channel` est recalculé sur les métriques stockées. On compare, par canal, les critères déclenchés (`flat_pct`, `n_peaks`…), pas le texte, qui contient le seuil. Encadré : `flags: C3 (flat_pct) → none`. |

- **Nouvelle colonne additive `flag_thresholds`** dans `{id}_quality_metrics.tsv` (`flat_pct:3.5+bounds_pct:1+…`, donc aussi
  dans `quality_summary.tsv` et chez l'outil 0, qui partage `compute_channel_metrics`). **La règle reste celle des flags** :
  cette colonne sert à reprendre les seuils (phase 5) et à montrer l'ancien seuil dans l'encadré, elle ne déclenche rien.
- `update_acq_info` garde la fréquence de chaque fichier. Fréquence inconnue ou mixte (export défectueux) : compté comme
  différent, ce qui ne coûte qu'une question.
- **Limite acceptée** : un seuil modifié sans effet sur les flags ne refait rien, et le rapport garde l'ancien seuil dans
  son texte.
- **Non comparés**, et dits dans le SPEC : suffixe d'hypnogramme, stades custom, configuration de canaux. Ils ne sont pas
  enregistrés et ne touchent pas les flags du signal entier.
- `SPEC §6` : la décision « warning only » du 2026-10-16 est remplacée.

## Phase 2 : outil 7 (+ jumeau Curry)

Nouvelle fonction `preprocessing_lib.settings_differences(sidecar, current)`. Trois clés additives dans
`{id}_preprocessing_params.json` : `reference`, `manual_events`, `hypnogram_file`.

| Réglage | Règle |
|---|---|
| `epoch_length_s` | absent = 30 |
| `resample`, `filter` | valeurs demandées |
| `notch` | absent = non appliqué |
| Méthodes (`methods_run` sans `event`) | l'événement varie avec les annotations du fichier, il est comparé à la ligne Événements |
| Seuils d'amplitude, flat, gradient | seulement si la méthode a tourné des deux côtés |
| Seuils 1/f, plage du fit, `psd_smoothing` | seulement si le 1/f a tourné des deux côtés. `psd_smoothing` absent = désactivé |
| Événements | `event_rejection` (appliqué, types), `manual_events` |
| Canaux du participant | `channels` + `bad_channels` contre l'éditeur |
| Référence, hypnogramme | clés nouvelles : absentes dans un ancien sidecar = non comparées |

Pré-remplissage de l'éditeur, au Scan :

- Participant déjà traité dans le run choisi, avec sidecar : l'état chargé vient du sidecar. *Reset channels to defaults*
  revient à cet état.
- Autres participants : flags de l'outil 6, comme aujourd'hui. Le motif de l'outil 6 reste affiché sous les cases.
- Changement de run : l'état suit le nouveau run, sauf pour un participant modifié dans la session. Cela précise la règle
  actuelle du SPEC (*the channels ticked in the editor stay*).
- **Limite acceptée** : un nouveau flag de l'outil 6 ne décoche plus rien chez un participant déjà traité.

Run : l'alerte « mixed-method » est limitée au run choisi. Elle reste utile avec *Skip* décoché sur un sous-ensemble.

## Phase 3 : outil 9

Deux catégories, car les réglages n'ont pas le même coût :

| Catégorie | Réglages | Refaire = |
|---|---|---|
| Calcul | PSD, fit, lissage, bandes, mesures, comparés à `cfg_json` du `_psd_epoch.npz` | calcul complet |
| Agrégation et décision | stades, espace de moyennage, epochs minimum, tiers de nuit, exclusion des mauvais fits, contenu de la décision | *Re-aggregate only* |

- **Nouveau fichier additif** `{id}_spectral_settings.json`, à côté du `.npz`, écrit par `finish_participant` avant le
  marqueur `_spectral_stage.tsv` : réglages d'agrégation, source et empreinte de la décision, stades du participant.
- Pas de fausse alerte sur un widget sans effet : `bandwidth` en Welch, fenêtre et recouvrement en multitaper, `fit` si le
  fit est décoché, `smooth` s'il est désactivé. Bandes et mesures sont comparées comme ensembles, `n_jobs` jamais.
- `stages` est limité aux stades que le participant possède : un stade custom apparu ailleurs dans la base ne le rend pas
  différent.
- L'encadré distingue `to recompute` et `to re-aggregate`, et le bouton applique le bon chemin à chacun.
- Sortie ancienne sans le fichier : calcul comparé via le `.npz`, agrégation et décision non comparées (ℹ️).

## Phase 4 : outil 8bis et aval de l'outil 7

**Outil 8bis et encadré** : la comparaison, aujourd'hui dans sa boucle, passe dans une petite passe avant la boucle, sur le
modèle des autres outils. Elle a besoin du `_epoch_channel_rejection.tsv` de chaque participant déjà décidé (les méthodes et
stades réellement utilisés en dépendent), mais son Skip le lit déjà aujourd'hui. Pas de nouveau calcul, et peu de code : les
lignes de la boucle deviennent une fonction locale appelée aux deux endroits.

**Empreinte** : `SF.preproc_fingerprint(sidecar_path)` dans `study_files_lib`. C'est un SHA-256 court d'une **liste fixe**
de clés de réglages du sidecar, sans `sfreq_hz` ni clé informative. Une clé ajoutée plus tard ne décale donc pas les
empreintes. Refaire l'outil 7 avec les mêmes réglages donne la même empreinte : les flags sont les mêmes, rien n'est
périmé.

| Outil | Enregistre | Quand l'empreinte diffère |
|---|---|---|
| 8bis | colonne `preproc_fingerprint` du `_autoreject_decision.tsv`, ajoutée à `DECISION_SETTINGS` | une ligne de l'encadré, comme ses propres réglages, le Scan annonce combien |
| 8 | colonne `preproc_fingerprint` du `_manualreject_decision.tsv` | **avertit seulement** à côté du badge : un travail manuel n'est jamais refait à ta place |
| 9 | clé `preproc_fingerprint` de `_spectral_settings.json` | catégorie *calcul* de l'encadré, plus un ⚠️ au Scan si la décision choisie est elle-même périmée |

Valeur absente dans un ancien enregistrement : rien à comparer, aucune alerte et aucun retraitement massif après la mise à
jour. `settings_changed` de 8bis est adapté pour cette seule colonne.

Jumeaux : re-générer ceux des outils 8 et 8bis. L'outil 9 n'a pas de jumeau.

## Phase 5 : reprendre les réglages enregistrés (outils 6, 7, 8bis, 9)

Troisième bouton de l'encadré, *Use the recorded settings*. **Il ne lance rien** : il remet les widgets aux valeurs
enregistrées, dit ce qu'il a changé, et tu cliques de nouveau sur Run.

Cas type : rouvrir l'outil 7 pour 2 nouveaux participants d'une étude aux réglages non standard. Scan, Run, encadré,
*Use the recorded settings*, Run : plus de différence, les 2 nouveaux sont traités comme les autres, les anciens ne bougent
pas.

| Outil | Lu dans | Widgets remis |
|---|---|---|
| 6 | `signal_preparation`, `flag_thresholds` | rééchantillonnage, high-pass, les 5 seuils |
| 7 | sidecar | toute la Section 2 : rééchantillonnage, notch, filtre, longueur d'epoch, méthodes et seuils (stades custom compris), fit 1/f, lissage, événements, annotations manuelles, plus le suffixe d'hypnogramme |
| 8bis | `_autoreject_decision.tsv` | seuils en %, stades de référence, méthodes, types d'événements et leurs seuils |
| 9 | `cfg_json` et `_spectral_settings.json` | toute la Section 2 : PSD, fit, lissage, bandes, mesures, stades, agrégation |

- **Run aux réglages mélangés** : on prend le jeu de réglages **le plus fréquent** parmi les participants du run (à égalité,
  le premier dans l'ordre des participants), et la confirmation dit combien l'ont utilisé (`27 of 30`). Le choix du jeu est
  dans `participant_selection_lib`, la mise à jour des widgets dans chaque notebook.
- **Deux exceptions à cette règle**, parce qu'un enregistrement ne garde pas tout ce qui était coché :
  - 8bis et 9, stades : on prend l'**union**, car chaque enregistrement ne garde que les stades que le participant possède.
  - Outil 6, rééchantillonnage : on le reprend dès qu'une partie du run l'a utilisé, car `none` peut venir de la règle qui
    interdit de suréchantillonner un fichier déjà à la bonne fréquence.
- **Ce qui ne peut pas être repris est nommé**, et reste tel quel sur la page : seuils d'une table de l'outil 6 écrite avant
  la colonne, agrégation d'une sortie de l'outil 9 sans le nouveau fichier, type d'événement absent du menu (pas
  d'`event_remap.json` choisi). Le contrôle au Run suivant reste le garde-fou.
- **Non repris** : la référence, qui vient de la configuration de l'outil 2 (un fichier, pas un widget). Si elle diffère,
  l'encadré revient et le dit. Les canaux, eux, sont déjà repris au Scan (phase 2).
- **Pas de remplissage automatique au Scan** : les widgets changeraient sans que tu le demandes, et ce que tu viens de
  saisir serait effacé. Pas de bouton permanent non plus : celui-ci n'apparaît que lorsqu'une différence existe.
- Jumeau Curry de l'outil 6 : son générateur retire `thresh_bounds`, il retirera aussi sa ligne dans la reprise.

## Phase 6 : documentation

- `SPEC.md` : *Skip + cumulative-merge* (la règle commune), §6, §7, §8, §8bis, §9, *Order of the TODOs*, suppression de ce
  TODO.
- `CLAUDE.md` : un rappel court dans *Shared conventions*, renvoi au SPEC.
- `TESTS.md`, et les textes d'intro des pages concernées, dont celui de l'outil 8bis (*decided again, as the log says*).
- Ce plan est supprimé à la fin, par son propre commit.

## Ce qui change dans les sorties

Aucune valeur existante ne change. Uniquement des ajouts : 1 colonne dans les tables de l'outil 6 et de l'outil 0, 3 clés
dans le sidecar de l'outil 7, 1 colonne dans les enregistrements de décision 8 et 8bis (et donc leurs tables globales),
1 fichier JSON par participant dans l'outil 9. Le comparateur des goldens tolère les colonnes et les clés ajoutées. **Pas de
golden à régénérer** : je le vérifierai au premier run, et je te le dirai si ce n'est pas le cas.

## Tests

- **Nouveau `tests/test_settings_skip.py`**, sur données synthétiques, chaque scénario en **nouvelle session** pour
  couvrir le cas réel :
  - Outil 6 : même réglage = pas d'encadré. High-pass changé = encadré et aucun fichier modifié. *Keep* reconstruit
    seulement. *Reprocess* change l'étiquette. Un seuil relevé ne liste que le participant dont le flag change.
  - Outil 7 : seuil changé, encadré, les deux boutons. Canal décoché à la main conservé à la réouverture. Un run `_v2` ne
    perturbe pas le premier.
  - Outil 8bis : seuil changé, encadré, les deux boutons. Mêmes réglages = sauté.
  - Outil 9 : bandes changées = recalcul. Epochs minimum ou décision changés = re-agrégation, égale à un calcul complet.
  - Aval : outil 7 refait avec d'autres réglages = 8bis et 9 demandent, 8 avertit. Mêmes réglages ou ancien
    enregistrement = rien.
  - Reprise (les 4 outils) : run aux réglages non standard, nouvelle session aux valeurs par défaut, un participant de
    plus. Encadré, *Use the recorded settings*, plus de différence au Run suivant, le nouveau a les mêmes réglages que les
    autres et leurs fichiers n'ont pas bougé. Run mélangé : le jeu le plus fréquent. Ce qui n'est pas repris est nommé.
- **Tests unitaires** des fonctions `settings_differences`, de l'empreinte, du regroupement de l'encadré et du choix du jeu
  le plus fréquent.
- **Tests existants à ajuster** : `test_tool6_resampling_and_highpass`, `test_tool8bis_skip_decides_again_with_other_settings`,
  et ceux qui relancent avec d'autres réglages en attendant un Skip silencieux (je les listerai en les rencontrant).

**Niveau retenu** : `standard` sur les marqueurs des outils concernés à la fin de chaque phase (`tool6`, puis `tool7`,
`tool8`, `tool8bis`, `tool9`, plus `test_curry_twins`). Pour un `full` à la toute fin, je te reposerai la question à ce
moment-là.

## Limites assumées

- L'empreinte suit les **réglages**. Un EDF ou un hypnogramme remplacé sous le même nom n'est pas détecté.
- Le contenu de `event_remap.json` n'est pas comparé, seuls les types choisis le sont.
- En cliquant sur *Keep*, tu acceptes une base aux réglages mélangés. L'encadré et le log le disent, le reste suit les
  garde-fous existants (`Mixed signal preparations` de l'outil 6, colonne `n_part_{m}` de l'outil 7).
- Le « R will run » du sélecteur ne tient pas compte des réglages, car ils peuvent encore changer après le Scan. La
  réponse exacte arrive au clic sur Run.

## Découpage

Une phase = un `git add` et un message de commit proposé. Les phases 2 et 3 sont indépendantes entre elles une fois
l'encadré commun posé en phase 1. La phase 4 s'appuie sur la phase 3 pour le fichier de l'outil 9. La phase 5 s'appuie sur
tout ce qui précède, puisqu'elle relit ce que les phases 1 à 4 enregistrent.
