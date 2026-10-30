# Plan : harmonisation de l'affichage et du fonctionnement des outils Voila

**Statut** : brouillon à relire (2026-10-29). Rien n'est implémenté avant ta confirmation.
**Contexte** : SPEC *Planned modules → Display harmonization across the Voila tools*. Tes décisions du 2026-10-29 :
- **dans cette tâche** : A, l'apparence (séparateurs, titres, largeurs, boutons, couleurs, barres de progression) ;
  B, les intros (gabarit commun, contenu gardé, plus les corrections des outils 1 et 2 listées dans SPEC) ;
  et, en **phase 2**, le fonctionnement (même façon d'avancer dans la page partout) ;
- **plus tard (restent en TODO)** : C, la sélection de fichiers pré-pointée vers les fichiers attendus (avec la TODO
  *Companion suffixes chosen from the detected ones*) ; D, renommer les sorties / plusieurs configurations de
  l'outil 2 ;
- **un petit module partagé** pour le style ;
- **titres de section** : widgets dans les outils 1 à 4, cellules Markdown ailleurs, avec le même rendu. Le
  découpage des outils 1 à 4 en une cellule par section est une TODO de SPEC (ajoutée) ;
- **outil 2 renuméroté** (sans « 2bis ») ; l'enrichissement des voies de contexte (vEOG, EMG menton / bras /
  jambe, ECG) est une TODO de SPEC à part (ajoutée).

**Règle d'or : aucune sortie ne change** (tables, JSON, `.fif`, figures, rapports HTML). Les noms de variables des
widgets restent (les tests les pilotent par leur nom), ainsi que les textes que les tests recherchent (annexe).
Un message qui est aussi écrit dans un fichier (journal enregistré, rapport) n'est pas touché.

---

## 1. État des lieux (audit du 2026-10-29)

| Outil | Titre actuel | Titres de section | Entrée | Barre de progression |
|---|---|---|---|---|
| 0 | Tool 0: one participant, the whole pipeline | Markdown sans numéro, puis onglets | Load the recording | non (étapes) |
| 1 | Inspect export parameters from .edf files… | widgets : trait noir épais + `1.`, trait fin + `2.1` | Run inspection | non |
| 1bis | `## EDF anonymizer` (pas de titre `#`) | `<h3>` avec émoji (📁 ⚙ 📝 ▶) | ▶ Run anonymization | non |
| 2 | Select, rename and define a re-reference… | widgets, trait épais, 1, 2, 2bis, 3…6 | Run scan + 5 boutons « Run … » qui ne font qu'afficher | non |
| 3 | Remap hypnogram labels | widgets, trait épais, 1…5 | Scan for hypnograms + 4 boutons d'affichage | non |
| 3bis | Tool 3bis: Automatic sleep scoring… | Markdown `## 1.` | Scan (bleu clair `info`) | oui |
| 4 | Harmonize scored-event labels… | widgets, 1, 1bis, 2…5 | Run scan | non |
| 5 | Tool 5: Sleep macrostructure… | Markdown `## 1.` | Scan | oui (2 barres) |
| 5bis | Tool 5bis: Participant selection… | Markdown `## 1.` | Load | sans objet |
| 6 | `## Preprocessing, Phase 1: Quality Overview` | `<h3>` avec émoji | ▶ Run (direct) | oui |
| 7 | Preprocessing: Phase 2 | Markdown `## 1. Paths`, `<h3>` + trait gris en section 2 | Load participants | oui |
| 8 | Tool 8: … (Phase 2b) | Markdown `## Section 1:` | Load participant | oui (ajustement 1/f) |
| 8bis | Tool 8bis: … | Markdown `## Section 1 -` | Scan | oui |
| 9 | Tool 9: … | Markdown `## 1.` | Scan | oui |

- **Couleurs** : 6 rouges (`#c62828`, `#c0392b`, `#c33`, `#dc3545`, `#d03b3b`, `#b71c1c`), 9 oranges et 6 verts
  pour les trois mêmes sens (ok / attention / erreur).
- **Boutons** : « Scan » est bleu (`primary`) dans 5, 8bis, 9, bleu clair (`info`) dans 3bis, vert (`success`)
  dans 2 et 4 ; « Run » est vert dans 1 à 5, 3bis, 8bis, 9 mais bleu dans 6 et 7. Certains libellés commencent
  par le caractère `▶`, d'autres utilisent l'icône `play`.
- **Fonctionnement** : dans les outils 2 et 3, chaque section a un bouton « Run … » qui ne sert qu'à faire
  apparaître ses widgets (héritage de la version Jupyter) ; l'outil 4 construit déjà ses sections tout seul
  après le scan (c'est le modèle à suivre).
  ==> puisque le bouton "Run" de chaque section qui fait apparaitre les widget est un héritage des versions jupyter, on peut en effet suivre le model de 4. 

---

## 2. Le module partagé : `tools/ui_lib.py`

Le style de toutes les pages Voila, en un seul endroit : des **constantes** et de **petites fonctions**, sans calcul
ni état. C'est une exception assumée à la règle « une bibliothèque ne crée pas de widget » de SPEC (qui vise les
bibliothèques de calcul) : `ui_lib` ne fait que l'apparence. Importé par chaque outil Voila (sonde `tools/`
habituelle ; les outils 2 et 4 et leurs jumeaux Curry n'en ont pas encore, elle est ajoutée), et par
`participant_selection_lib` pour les couleurs de l'écran du sélecteur (pas celles des rapports).

### 2.1 Couleurs des messages (palette « Material », déjà la plus utilisée : outils 0, 5, 8, 9)

| Sens | Texte | Fond de boîte | Symbole | Contraste sur blanc |
|---|---|---|---|---|
| ok, terminé | `#2e7d32` | `#e8f5e9` | ✅ | 5.1:1 |
| attention | `#e65100` | `#fff3e0` | ⚠ | 3.8:1 (voir question 1) |
| erreur | `#c62828` | `#ffebee` | ❌ | 5.6:1 |
| information | `#1565c0` | `#e3f2fd` | ℹ | 5.8:1 |
| discret (aide, détail) | `#666` | | | 5.7:1 |

Le rouge d'erreur `#c62828` reste distinct du rouge du REM `#c0392b` des figures (*Display conventions*).
Fonctions : `msg(kind, text)` (une ligne colorée avec son symbole) et `box(kind, html)` (une boîte : bordure gauche
de 4 px, fond pâle, texte sombre ; pour les avertissements longs et les bilans de fin de run).

### 2.2 Titres de section

- `section_html(n, title, intro='')` : le trait noir épais des outils 1 à 3, puis `N. Titre`, puis le texte
  d'explication de la section (outils à widgets : 1, 1bis, 2, 3, 4, 6).
- `subsection_html(n, title, intro='')` : trait noir fin, puis `N.M Titre`.
- Dans les outils à cellules Markdown, la cellule de section reçoit le même trait (la même ligne HTML) au-dessus de
  `## N. Titre`. Un test vérifie qu'elle est identique à celle de `ui_lib` (pas de dérive).

### 2.3 Boutons : un style par rôle

`button(description, role, icon=None)` crée le bouton ; la variable reste nommée dans le notebook.

| Rôle | Exemples | Style | Icône par défaut |
|---|---|---|---|
| `run` : l'action principale, qui calcule ou écrit | Run, Run the inspection, Anonymize, Compare the models | `success` (vert) | `play` |
| `save` : enregistrer un choix ou un fichier | Save selection, Save remapping, Save decision | `success` (vert) | `save` |
| `read` : lire ou lister sans rien écrire | Scan, Load, Load participant, Preview, Count affected epochs | `primary` (bleu) | `search` / `folder-open` / `eye` |
| `caution` : change beaucoup d'éléments d'un coup, ou exclut | Apply to all shown, Apply to all in this file, Exclude this participant | `warning` (orange) | selon le cas |
| `helper` : petits boutons | ◀ Prev, Next ▶, Tick all, Add band, corbeille | gris | selon le cas |

Plus de style `info`, plus de `▶` dans le texte (l'icône `play` le remplace). Largeur `auto` (le bouton prend la
largeur de son texte), sauf dans une rangée de boutons de même rôle.

### 2.4 Largeurs et champs

- **Sélecteurs de dossier / fichier** (`FileChooser`) : même largeur partout, titre en gras (`<b>Study folder</b>`)
  et l'aide en gris dans une ligne en dessous, pas dans le titre.
- **Champs texte / listes / nombres** : étiquette de largeur fixe (`LABEL_W`, environ 170 px) pour que les champs
  d'une section s'alignent ; trois largeurs de champ : `W_FIELD` (suffixes, noms de fichier, ~460 px), `W_SHORT`
  (nombres, seuils, ~260 px), `W_WIDE` (listes longues, ~620 px). Valeurs fixées sur la page de démonstration
  (étape 1).
- **Journaux de run** (`Output`) : même cadre partout (bordure grise fine, hauteur maximale + défilement).
- **Ce qui ne bouge pas** : les panneaux propres à un outil, dessinés exprès : les seuils de l'outil 7 (retraits
  « Variant A »), le navigateur de l'outil 8, l'explorateur et les onglets de l'outil 0, les panneaux par
  configuration de l'outil 2, le sélecteur de participants partagé (seulement ses couleurs).

### 2.5 Barre de progression

`progress_bar(description='Participants:')` : une barre et un texte `i / N` à droite ; bleue pendant le run,
verte à la fin sans échec, orange s'il y a eu des échecs. Une seule forme pour tous les outils (la barre
« étape » des outils 5, 7, 9 garde sa place, sous la première).

---

## 3. Phase 1 : l'apparence et les intros (A + B)

### 3.1 Gabarit de la cellule d'intro (Markdown)

```
# Tool N: <titre>

<Une ou deux phrases : ce que fait l'outil.>

**Inputs**: …
**Outputs**: … (dans quel dossier)
**Sections**: 1. …, 2. …, 3. …

<details><summary>Details</summary>
… le reste du texte actuel, gardé …
</details>
```

==> si possible il faudrait rajouter les ref correspondant aux outils, soit dans une dernière sous-section de l'intro ou alors dans une section à la fin du notebook

Le bloc repliable (`<details>`) garde les longs textes sans allonger la page (contexte EDF de l'outil 1, les
modèles de 3bis, l'arborescence des sorties de l'outil 7, les règles de 8bis…) : voir question 4.
==> si possible il faudrait rajouter les ref correspondant aux outils, soit dans ce bloc details de l'intro ou alors dans une section à la fin du notebook

**Titres proposés** (les « Phase 1 / 2 / 2b » disparaissent des titres) :

| Outil | Titre |
|---|---|
| 0 | Tool 0: One participant, the whole pipeline |
| 1 | Tool 1: Inspect the EDF export parameters |
| 1bis | Tool 1bis: Anonymize EDF headers |
| 2 | Tool 2: Select, rename and re-reference the channels |
| 3 | Tool 3: Remap the hypnogram labels |
| 3bis | Tool 3bis: Automatic sleep scoring (GSSC, YASA, SleepFM) |
| 4 | Tool 4: Harmonize the scored-event labels |
| 5 | Tool 5: Sleep macrostructure |
| 5bis | Tool 5bis: Participant selection on sleep macrostructure criteria |
| 6 | Tool 6: Signal quality overview |
| 7 | Tool 7: Preprocessing and epoch flagging |
| 8 | Tool 8: Manual epoch rejection |
| 8bis | Tool 8bis: Automatic epoch rejection (channel-first, then epoch) |
| 9 | Tool 9: Spectral features (band power, aperiodic components) |
| jumeaux Curry | même titre + « (Curry 9 .cdt) » |

**Corrections listées dans SPEC** :
- outils 1 et 2 : les noms internes (« developed on the ICEBERG database… », « …APOMORPHEE… ») remplacés par
  les organisations de dossiers reconnues (« Developed and tested on PSG databases exported from Compumedics
  Profusion, with groups as subfolders or nights as file-name suffixes ») ; les lignes « last update …, YN »
  supprimées ; mêmes noms internes dans les commentaires du code de l'outil 1 (Voila + les deux scripts batch, des
  commentaires seulement) et de l'outil 2 ;
- outil 1 : le lien manquant « (insert a link on how to make a routine in Compumedics) » pointe vers
  `tutorials/2_create_EEG_montage_Profusion.docx` ;
- outil 1 : « It is a compressed 16-bit format » → l'EDF n'est pas compressé : chaque échantillon est un entier
  de 16 bits, mis à l'échelle entre le minimum et le maximum physiques fixés à l'export ;
- outil 2 : la coquille « dispaly ».

**Phrases d'avant Voila à retirer** : outil 2 « reduce it by pressing the down arrow on the left of the section
title » (n'existe pas dans Voila), outil 3 « Run all cells (Kernel → Restart & Run All) », outil 1 « run
`voila tools/1_inspect_edf_voila.ipynb`… » (la page est déjà ouverte), la cellule de l'outil 1 « First, let's
check if you have already installed the required packages » (la vérification reste, silencieuse quand tout va
bien, comme dans les autres outils ; son message d'échec garde le texte `Import error`).

### 3.2 Les sections

- **Même titre partout** : trait noir épais + `N. Titre` ; sous-section : trait fin + `N.M Titre`.
- **Mêmes noms quand le rôle est le même** : `1. Study folder` (ou `1. Folders` quand la section a plusieurs
  sélecteurs : 7, 8, 8bis, 9), `N. Parameters`, `N. Participants`, `N. Run`, et en fin de page, sans numéro,
  `Outputs: where to look` (5, 5bis l'ont ; le « Once the run is finished » de l'outil 9 prend ce nom).
- **Dans chaque section, toujours le même ordre** : les choix (dossiers, paramètres), puis un bouton d'action, puis
  la ligne d'info ou le journal.
- `Section 1:` (outil 8) et `Section 1 -` (8bis) deviennent `1.`.
- Outil 6 : `1. Study folder`, `2. Signal preparation`, `3. Quality thresholds`, `4. Run` (aujourd'hui des
  `<h3>` à émoji). Outil 1bis : `1. Original EDF folder`, `2. Options`, `3. Files to anonymize`, `4. Run`.
- Outil 7, section 2 : ses quatre blocs (Preprocessing, Epoching, Epoch flagging, Events) deviennent les
  sous-sections 2.1 à 2.4.
- Outil 0 : `1. Study folder and participant`, `2. Steps`, `3. Epoch explorer` ; ses onglets et accordéons
  restent.

### 3.3 Messages, boutons, largeurs, barres

- Toutes les couleurs écrites en dur **à l'écran** passent par `ui_lib` (`msg`, `box`, constantes) ; le texte des
  messages ne change pas.
- Les journaux `print()` à l'écran (non colorables) : symboles unifiés ✅ ⚠ ❌ (voir question 2).
- Boutons créés avec `ui_lib.button` selon leur rôle (§2.3) ; libellés inchangés en phase 1.
- Champs et sélecteurs aux largeurs de §2.4 (hors panneaux propres à un outil).
- Barres de progression existantes (3bis, 5, 6, 7, 8, 8bis, 9) mises à la forme de §2.5.

---

## 4. Phase 2 : le fonctionnement (toujours sans changer de sortie)

### 4.1 Plus de boutons qui ne font qu'afficher (outils 2 et 3)

Une section se construit d'elle-même dès que l'étape précédente est faite. Avant, elle montre une ligne grise
« Complete section N first. ». Quand un choix en amont change, la fonction `invalidate_after` (qui existe déjà
dans les deux outils) remet cette ligne au lieu de réafficher le bouton. Les vrais boutons d'action restent.

**Outil 2** (renuméroté) :

| Section | Action réelle | Construit ensuite |
|---|---|---|
| 1. Study folder | Scan | 2, 3 et 7 |
| 2. Select channels of interest | Save selection | 4 |
| 3. Identify context channels (EOG L/R, EMG, ECG), optionnelle (ex-2bis) | Save context selection | |
| 4. Rename channels | Save remapping | 5 |
| 5. Define the re-reference method | Save re-reference | 6 |
| 6. Preview and save the JSON file | Preview & Save | |
| 7. Test the JSON file | Test the JSON (lit le fichier sur le disque, donc dispo dès le scan, comme aujourd'hui) | |

Supprimés : « Run context channels », « Run channel remapping », « Run re-reference method », « Run JSON
saving », « Run test JSON file » (ce dernier devient le vrai bouton de la section 7).

**Outil 3** :

| Section | Action réelle | Construit ensuite |
|---|---|---|
| 1. Study folder | Scan | 2 (la revue s'ouvre directement s'il y a des époques à revoir ; sinon « Nothing to review » et 3 se construit) |
| 2. Review uncertain and unexpected epochs | Corrections done: proceed (reste : il écrit le TSV) | 3 |
| 3. Define remapping rules | Save remapping | 4 |
| 4. Preview and save remapped hypnograms | Preview, Save files | 5 |
| 5. Verify remapping | Verify (vrai bouton : relit les fichiers écrits) | |

Supprimés : « Review uncertain & unexpected epochs », « Run label remapping widget », « Run preview & save » ;
« Run verification » devient « Verify » (variable `btn_run_verify` gardée).

**Autres outils** : l'outil 4 suit déjà ce modèle. À vérifier pendant l'implémentation : le « Start navigator » de
l'outil 8 ; s'il ne fait qu'afficher, même traitement.

### 4.2 Numérotation continue

- **Outil 2** : 1 à 7 (ci-dessus). Renvois mis à jour : SPEC §2 (Sections 2bis, 4, 5, 6), la puce
  *Context-channel detection*, les deux nouvelles TODO, CLAUDE.md (« Section 2bis »), deux textes affichés par
  l'outil 3bis (« tool 2's Section 2bis »), un commentaire de `edf_io_lib.py`, et les étiquettes du générateur
  `_make_tool2_curry.py` (« Section 6 … » → 7).
- **Outil 4** (question 3) : la section « 1bis. (Optional) Check text/CSV vs XML consistency » devient une
  sous-section **sans numéro** à la fin de la section 1 (« Optional: check text/CSV vs XML consistency »). Le
  jumeau Curry, qui supprime ce bloc, garde ainsi 1 à 5 sans trou.

### 4.3 Même entrée : la section 1

- Premier sélecteur intitulé **Study folder** partout (déjà la règle), avec la même ligne d'aide.
- La case *Skip already processed* et la ligne « N / M already … » toujours au même endroit, juste sous le
  sélecteur, dans le même style.
- Le bouton qui liste les participants avant de lancer s'appelle **Scan** (rôle `read`) : outils 2 (« Run
  scan »), 3 (« Scan for hypnograms »), 4 (« Run scan »), 7 (« Load participants ») ; 3bis, 5, 8bis, 9 l'ont
  déjà. Restent : « Load » (5bis, charge une table), « Load participant » (8, un participant à la fois), « Run
  inspection » (1, son unique action) et l'outil 6 (pas d'étape de liste, il lance directement).

### 4.4 Barres de progression ajoutées

Outils 1 (inspection), 1bis (anonymisation), 2 (scan), 3 (scan), 4 (scan) : la barre de §2.5.

### 4.5 Libellés des boutons

Verbe + objet, phrase courte, majuscule seulement au début. La liste exacte des changements sera dans le
commit ; les libellés que les tests cliquent restent (« Save remapping », « Save files », « Exclude this
participant » / « Remove my exclusion »).

---

## 5. Jumeaux Curry

- **Générés** (2, 3, 4, 6, 7, 8, 8bis) : relancer le générateur après chaque notebook EDF. Chaînes à adapter là
  où le texte modifié est recherché : outil 3 (titre), outil 4 (intro, bandeau de la section 1, suppression de la
  1bis), outil 6 (cellule de titre), outil 7 (titre), outil 8 (titre + une phrase), outil 2 (titre du sélecteur,
  texte d'info de la section 6 → 7). Les générateurs 2 et 4 ajoutent la sonde `tools/` pour `ui_lib`.
- **Outil 1** (jumeau écrit à la main) : mêmes changements, à la main.
- `test_curry_twins.py` : `ui_lib` ajouté aux marqueurs.

---

## 6. Tests

- **Avant de commencer** (code de confiance) : `tests/refactor_check.py save before_display` sur les deux
  jeux de données, plus `tests/outside_chain.py 1 3 3bis 6variant 7variant`. À la fin de chaque phase,
  `compare` doit répondre « no difference ».
- **Nouveau test rapide `tests/test_ui_lib.py`** : les fonctions de `ui_lib` rendent le HTML attendu ; garde-fou
  contre la dérive : chaque notebook Voila (et chaque jumeau) importe `ui_lib`, n'utilise plus le style de bouton
  `info`, et ses cellules Markdown de section portent le même trait que `section_html`.
- `tests/chaincache.TOOL_DEPS` : `tools/ui_lib.py` ajouté aux outils 5 à 9 (une modification relance la chaîne).
- **Tests à adapter en phase 2** : les clics de l'outil 3 sur `btn_run_remap` / `btn_run_save`
  (`test_tools_1_3_4.py`, `outside_chain.py`).
- **Niveaux** (je te demanderai à chaque fois) : rapide après chaque outil (le notebook s'exécute + jumeaux) ;
  complet + `refactor_check compare` à la fin de chaque phase (un module partagé touche tous les outils).
- **Vérification visuelle par toi dans Voila** (aucun test ne voit le rendu) : la page de démonstration à
  l'étape 1, puis chaque outil à la fin de chaque phase, avec une courte liste de points à regarder.
- `TESTS.md` mis à jour.

---

## 7. Documentation

- **SPEC** : une nouvelle puce *Cross-cutting → Voila page conventions (`ui_lib`)* (gabarit d'intro, titres de
  section, palette, rôles des boutons, largeurs, barre de progression, pas de bouton qui ne fait qu'afficher,
  numérotation continue) ; l'arborescence et le tableau des bibliothèques (`ui_lib`, importé par chaque outil Voila
  et par `participant_selection_lib`) ; §2 (numéros), §3 (déroulement), §4 (sous-section optionnelle) ; l'entrée
  *Display harmonization* remplacée par deux TODO restantes, C (sélection de fichiers pré-pointée) et D (renommer
  les sorties / plusieurs configurations de l'outil 2) ; les deux nouvelles TODO renvoient à la section 3 de
  l'outil 2.
- **CLAUDE.md** : un rappel court dans *Shared conventions* (« Voila page conventions = `tools/ui_lib.py` ; une
  nouvelle page ou un nouveau widget la suit → SPEC ») et « Section 2bis » → « Section 3 ».
- `README.md` : vérifier s'il cite des numéros de section ou des libellés de boutons.

---

## 8. Ordre de travail

| Étape | Contenu | Fin d'étape |
|---|---|---|
| 0 | `refactor_check save` + `outside_chain` sur le code actuel | |
| 1 | `ui_lib.py` + `test_ui_lib.py` + une **page de démonstration** (dans le scratchpad, non versionnée) : titre Markdown et titre widget côte à côte, les cinq types de message, les cinq rôles de bouton, une barre de progression, un sélecteur + des champs | **tu l'ouvres dans Voila et valides le rendu** |
| 2 | Phase 1, outil par outil : 1, 1bis, 2, 3, 4, puis 3bis, 5, 5bis, 6, 7, 8, 8bis, 9, 0 ; pour chacun : intro, titres, couleurs, boutons, largeurs, barres, son jumeau Curry, test rapide | commits par groupe (1 à 4 / 3bis à 5bis / 6 à 9 / 0) |
| 3 | Fin de phase 1 : tests complets + `refactor_check compare` + ta vérification visuelle | commit |
| 4 | Phase 2 : outils 2 et 3 d'abord (les plus gros), puis l'entrée et les barres de 1, 1bis, 4, 7, puis les libellés ; tests adaptés | tests complets + `compare` + vérification visuelle, commit |
| 5 | Documentation (SPEC, CLAUDE.md, TESTS.md) ; suppression de ce plan | deux commits (doc, puis suppression du plan) |

---

## 9. Questions ouvertes (annote directement ici)

1. **Orange d'attention** : `#e65100` (déjà utilisé, contraste 3.8:1 sur blanc, assez pour du gras ou une
   bordure) ou un orange plus sombre, `#bf4b00` (5:1, plus lisible en texte courant mais plus « brun ») ?
   ==> `#e65100` je préfère celle la
2. **Symboles des journaux `print()`** : unifier aussi les journaux à l'écran en ✅ ⚠ ❌ (aujourd'hui ⚠ et ⚠️,
   ✅ ✓ ✔, ❌ ✗) ? Je le recommande ; un journal qui est aussi écrit dans un fichier n'est pas touché.
   ==> ok pour unifier sur ✅ ⚠️ ❌
3. **Outil 4, section 1bis** : sous-section optionnelle sans numéro en fin de section 1 (recommandé), ou une vraie
   section 2 avec renumérotation 3 à 6 (le générateur Curry renumérote alors à son tour) ?
   ==> ok pour sous-section optionnelle sans numéro en fin de section 1
4. **Longs textes d'intro dans un bloc repliable** `<details>` (recommandé, la page commence par l'essentiel)
   ou entièrement visibles ?
   ==> ok pour une section details repliées
5. **Titres** du tableau §3.1 : d'accord, ou à reformuler ?
==> ok avec les titres proposés
6. **« Scan »** pour l'entrée des outils 2, 3, 4 et 7 (l'outil 7 perd « Load participants ») : d'accord ?
==> ok pour Scan

---

## Annexe : textes que les tests recherchent (à garder)

`Import error`, `Traceback`, `Save remapping`, `Save files`, `Remove my exclusion`, `Give a reason`, `EXCLUDED`,
`not excluded`, `needs a comment`, `Exclusion registry updated`, `deleted`, `<b>3</b> kept / 4` (outil 5bis),
`No participant to process`, `not the study folder`, `0</b> will run` et `will NOT be processed` (sélecteur
partagé), `not saved` (outil 0), `Same decision as tool 8bis`, `Re-aggregate only`, `1 processed`, `failed:`,
`⚠ made with resampled to 128 Hz, high-pass 0.5 Hz, this run: none` (outil 6), `chosen by hand` (journal écrit de
l'outil 0).
