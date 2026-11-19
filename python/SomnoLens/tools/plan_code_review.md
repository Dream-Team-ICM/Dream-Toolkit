# Plan : revue de code du 2026-10-24 (points #10, #8 / #16, #20)

**Statut** : version 1 du 2026-11-19, à confirmer.
**Source** : SPEC.md, *Planned modules → Code review of 2026-10-24* (point 1 de *Order of the TODOs*), d'après
`python/Score_EDF/DreamToolkit_Claude_Report.md`. Le #11 a été vérifié le 2026-10-28 : rien à changer.

## Décisions prises (2026-11-19)

1. **Outil 5** : un fichier d'événements présent mais non lu devient une ligne `warning` de sa table des checks
   (`event_file`), donc dans le checks TSV et le rapport HTML du participant, comptée dans `n_warnings`.
2. **Outil 4** : une ligne ⚠ par fichier non lu dans le log du Scan, et la vraie raison dans `failed_event_read.tsv`
   quand aucune source n'est lisible.
3. **#20** : les champs par voie nommés avec leur largeur, les décalages en découlent. Même lecture octet par octet.

## Ce qui reste muet aujourd'hui

`edf_io_lib.load_events` (outils 0 et 5) et sa copie de l'outil 7 ont reçu un « FIX #10 » qui écrit les exceptions dans
`logging` : sous Voila personne ne les voit. Et plusieurs cas ne passent même pas par là :
- un TXT présent mais sans heure de début d'enregistrement (BrainVision en pause, date d'en-tête illisible) ;
- un TXT ou un CSV lu sans aucune ligne d'événement (format inattendu, CSV sans colonnes Name / Start / Duration) ;
- un XML sans `<ScoredEvent>` ;
- un `_manual_events.tsv` illisible (`except Exception: pass`).

Dans tous ces cas les outils passent à la source suivante, ou disent *No event companion found* alors qu'un fichier
existe. L'outil 4 a son propre `load_events` (les noms seulement), avec trois `except Exception: pass`.

## Phase 1 : `load_events` dit quel fichier il n'a pas lu et pourquoi (#10)

**`edf_io_lib.load_events` et la copie de l'outil 7** (identiques, `tests/test_edf_io_sync.py`) :
- un argument additionnel `warn=None`, comme `preprocessing_lib.epoch_psd_for_1f(…, warn=)` ;
- chaque fichier trouvé mais non retenu donne sa raison : *not read: the recording start is unknown (or the recording
  was paused)*, *unreadable (UnicodeDecodeError: …)*, *no event line in the expected format*, *no Name / Start /
  Duration columns*, *no <ScoredEvent>* ;
- **une seule ligne par enregistrement**, qui finit par ce qui a été fait : par exemple
  `73_ScoredEvents_Export.txt unreadable (UnicodeDecodeError: …): events read from 73_event_xml.csv instead.` ou
  `…: no scored event read.` ;
- `_manual_events.tsv` illisible : *unreadable (…): the manual annotations are left out* (un fichier vide, après la
  suppression de toutes les annotations, reste muet : ce n'est pas une erreur) ;
- sans `warn` (scripts, tests) la ligne part dans `logging` comme aujourd'hui ;
- **la valeur rendue ne change pas** (`(df, source)` ou `(None, None)`) : aucun appel ni test existant ne bouge, et
  quand tous les fichiers sont lisibles il n'y a aucun message.

**Les appels** :
- **outil 7**, Run : la ligne dans le log du participant, `[id] ⚠ …`, juste avant l'actuel *No event companion found*
  quand rien n'est lu ; *Count affected epochs* : les lignes sous l'estimation ;
- **outil 0** : ligne de chargement (étape 2) et étape 6, dans ses `notes` ;
- **outil 5** (`macrostructure_lib.compute_recording`) : un check `event_file`, statut `warning`, le détail = la ligne
  (décision 1).

Les commentaires `# FIX #8` et `# FIX #10` disparaissent (le code dit ce qu'il fait).

## Phase 2 : outil 4 (#10 et #8 / #16)

- `load_events_from_txt` lit avec `Path(txt_path).read_bytes()` (#8 / #16).
- Son `load_events` prend le même `warn=None` et les mêmes raisons.
- **Changement de comportement voulu** : un TXT ou un CSV lu sans aucun événement passe à la source suivante, comme
  dans `edf_io_lib`. Aujourd'hui l'outil 4 garde la liste vide (0 label), alors que les outils 5 et 7 lisent la source
  suivante : leurs labels pouvaient manquer à `event_remap.json` sans que personne le voie. Sur des fichiers normaux
  rien ne change.
- Scan : une ligne `⚠ [id] …` par enregistrement concerné (décision 2), et quand aucune source n'est lisible la raison
  écrite dans `failed_event_read.tsv` est cette ligne au lieu de *no readable event companion*.
- Contrôle 1bis : le statut `error: <Type>` devient `error: <Type>: <message>`.

## Phase 3 : `read_edf_header_info` (#20)

Une liste des champs par voie avec leur largeur (label 16, transducer 80, dimension 8, physical min / max 8, digital
min / max 8, prefiltering 80, samples per record 8, reserved 32), les décalages calculés en cumulant. Mêmes octets lus,
même décodage `latin-1`, mêmes tolérances : sortie identique par construction.

## Tests

- **quick, nouveau** `tests/test_event_sources.py` : sur un petit enregistrement synthétique et des compagnons écrits
  pour le test, chaque cas de la phase 1 (TXT illisible + CSV bon → `csv` et une ligne qui nomme le TXT ; CSV sans
  colonnes + XML bon → `xml` ; tout cassé → `(None, None)` et *no scored event read* ; manuel illisible ; tout lisible
  → aucun message ; sans `warn` → `logging`), le même jeu sur le `load_events` de l'outil 4 (extrait du notebook), et
  `read_edf_header_info` comparé à `read_edf_header_custom` (labels, dimensions, échantillons par enregistrement).
- `test_edf_io_sync.py` garde la copie de l'outil 7 identique.
- **Preuve que rien ne change** : `tests/refactor_check.py` sauvé **avant** la phase 1 sur la chaîne synthétique,
  comparé après (aucun compagnon cassé dans la chaîne : toutes les tables identiques).
- Niveau proposé au moment de lancer (outils 4, 5, 7, 0).

## Documentation

- SPEC : *Event sourcing* (l'argument `warn`, les messages, le repli de l'outil 4), §4 (Scan, 1bis), §5 (le check
  `event_file`), §7 (la ligne du log et la phrase sur *FIX #8 / FIX #10*), §0 ; la section *Code review of 2026-10-24*
  fermée ; *Order of the TODOs* : le point 1 fait, le suivant devient *Tool 7: band-pass phase* + *Lighter conda
  environments*.
- CLAUDE.md : rien (aucune règle transversale ne change).
- Ce plan supprimé à la fin, dans son propre commit.
