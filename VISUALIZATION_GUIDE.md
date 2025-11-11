# Οδηγός Διαδραστικών Visualizations

Αυτός ο οδηγός εξηγεί πώς να χρησιμοποιήσετε τα διαδραστικά εργαλεία visualization για το project Mountain Car PPO.

## 📊 Διαθέσιμα Visualizations

### 1. **Training Dashboard** (`visualize_training.py`)

Δημιουργεί διαδραστικά HTML dashboards που δείχνουν:
- **Training Progress**: Rewards και lengths επεισοδίων σε πραγματικό χρόνο
- **Hyperparameters**: Visualization διαμόρφωσης μοντέλου
- **Agent Performance**: Πώς ο πράκτορας αποδίδει στα επεισόδια
- **Summary Dashboard**: Περιεκτικά στατιστικά

### 2. **Real-time Agent Viewer** (`view_agent.py`)

Παρακολουθήστε τον εκπαιδευμένο πράκτορα να παίζει Mountain Car σε πραγματικό χρόνο με οπτικό παράθυρο.

## 🚀 Γρήγορη Έναρξη

### Βήμα 1: Εκπαίδευση Μοντέλου (αν δεν έχει γίνει ήδη)

```bash
python mountain_car_ppo.py
```

Αυτό θα:
- Εκπαιδεύσει τον πράκτορα PPO
- Αποθηκεύσει training metrics στο `./training_logs/training_metrics.json`
- Δημιουργήσει αυτόματα visualizations στο τέλος

### Βήμα 2: Δημιουργία Διαδραστικών Dashboards

Μετά την ολοκλήρωση της εκπαίδευσης, εκτελέστε:

```bash
python visualize_training.py
```

Αυτό δημιουργεί HTML αρχεία στον φάκελο `./visualizations/`:
- `main_dashboard.html` - Κύριο entry point με συνδέσμους σε όλα τα dashboards
- `training_dashboard.html` - Πρόοδος εκπαίδευσης και learning curves
- `hyperparameters.html` - Visualization hyperparameters μοντέλου
- `agent_performance.html` - Ανάλυση συμπεριφοράς πράκτορα
- `summary_dashboard.html` - Περιεκτική περίληψη

### Βήμα 3: Προβολή Dashboards

Ανοίξτε το `./visualizations/main_dashboard.html` στον web browser σας για πρόσβαση σε όλα τα διαδραστικά visualizations!

**Χαρακτηριστικά:**
- ✨ Διαδραστικά plots (zoom, pan, hover για λεπτομέρειες)
- 📈 Visualization δεδομένων σε πραγματικό χρόνο
- 🎯 Μετρικές απόδοσης
- ⚙️ Εμφάνιση hyperparameters

### Βήμα 4: Παρακολούθηση Παιχνιδιού Πράκτορα

Για να δείτε τον πράκτορα σε δράση:

```bash
python view_agent.py
```

**Επιλογές:**
```bash
# Προβολή 5 επεισοδίων (προεπιλογή)
python view_agent.py

# Προβολή συγκεκριμένου αριθμού επεισοδίων
python view_agent.py ./models/mountain_car_ppo_final.zip 10

# Έλεγχος ταχύτητας αναπαραγωγής (δευτερόλεπτα ανά βήμα)
python view_agent.py ./models/mountain_car_ppo_final.zip 5 0.1
```

## 📁 Δομή Αρχείων

```
Mountain Car RL Algorithm/
├── mountain_car_ppo.py          # Κύριο training script (με monitoring)
├── visualize_training.py        # Generator διαδραστικών dashboards
├── view_agent.py                # Real-time agent viewer
├── training_logs/
│   └── training_metrics.json   # Training metrics (αυτόματα δημιουργημένο)
├── visualizations/
│   ├── main_dashboard.html      # Κύριο entry point dashboard
│   ├── training_dashboard.html  # Πρόοδος εκπαίδευσης
│   ├── hyperparameters.html    # Hyperparameters
│   ├── agent_performance.html  # Απόδοση πράκτορα
│   └── summary_dashboard.html  # Στατιστικά περίληψης
└── models/
    └── mountain_car_ppo_final.zip  # Εκπαιδευμένο μοντέλο
```

## 🎨 Χαρακτηριστικά Dashboard

### Training Dashboard
- **Episode Rewards**: Scatter plot που δείχνει reward ανά επεισόδιο
- **Episode Lengths**: Πόσο διήρκησε κάθε επεισόδιο
- **Mean Rewards**: 10-episode rolling average
- **Mean Lengths**: 10-episode rolling average
- **Διαδραστικό**: Hover για ακριβείς τιμές, zoom και pan

### Hyperparameters Dashboard
- Οπτικό bar chart όλων των PPO hyperparameters
- Δείχνει learning rate, batch size, discount factor, κ.λπ.
- Χρωματικά κωδικοποιημένο για εύκολη ανάγνωση

### Agent Performance Dashboard
- **Position Over Time**: Πώς αλλάζει η θέση του αυτοκινήτου
- **Velocity Over Time**: Αλλαγές ταχύτητας κατά τη διάρκεια επεισοδίου
- **Phase Space**: Trajectory Position vs Velocity
- **Actions Over Time**: Ποια actions επέλεξε ο πράκτορας
- Δείχνει πολλαπλά επεισόδια για σύγκριση

### Summary Dashboard
- Καμπύλη πρόοδου εκπαίδευσης
- Αποτελέσματα αξιολόγησης (box plots)
- Ιστόγραμμα κατανομής rewards
- Κατανομή μήκους επεισοδίων
- Στατιστικά επιτυχίας

## 🔄 Παρακολούθηση σε Πραγματικό Χρόνο

Κατά τη διάρκεια της εκπαίδευσης, τα metrics αποθηκεύονται αυτόματα στο `./training_logs/training_metrics.json`. Μπορείτε να:

1. **Παρακολουθήσετε κατά την εκπαίδευση**: Το αρχείο metrics ενημερώνεται κάθε 10 επεισόδια
2. **Δημιουργήσετε dashboards οποιαδήποτε στιγμή**: Εκτελέστε `visualize_training.py` ακόμα και ενώ η εκπαίδευση είναι σε εξέλιξη
3. **Προβάλετε στον browser**: Ανοίξτε τα HTML αρχεία - ενημερώνονται όταν κάνετε refresh

## 💡 Συμβουλές

1. **Καλύτερος Browser**: Χρησιμοποιήστε Chrome, Firefox, ή Edge για καλύτερη συμβατότητα
2. **Refresh Dashboards**: Αν η εκπαίδευση συνεχίζεται, κάντε refresh το HTML για να δείτε τα τελευταία δεδομένα
3. **Πολλαπλά Επεισόδια**: Προβάλετε τον πράκτορα με περισσότερα επεισόδια για να δείτε συνέπεια
4. **TensorBoard**: Ελέγξτε επίσης `tensorboard --logdir ./tensorboard_logs/` για επιπλέον metrics

## 🐛 Αντιμετώπιση Προβλημάτων

**Δεν δημιουργούνται visualizations?**
- Βεβαιωθείτε ότι η εκπαίδευση ολοκληρώθηκε επιτυχώς
- Ελέγξτε ότι το `./training_logs/training_metrics.json` υπάρχει
- Επαληθεύστε ότι τα `plotly` και `pandas` είναι εγκατεστημένα: `pip install plotly pandas`

**Ο agent viewer δεν λειτουργεί?**
- Βεβαιωθείτε ότι το αρχείο μοντέλου υπάρχει: `./models/mountain_car_ppo_final.zip`
- Ελέγξτε ότι το `gymnasium` είναι εγκατεστημένο με rendering support
- Σε ορισμένα συστήματα, μπορεί να χρειαστείτε: `pip install pyglet`

**Τα dashboards δεν είναι διαδραστικά?**
- Βεβαιωθείτε ότι ανοίγετε τα HTML αρχεία σε web browser (όχι text editor)
- Ελέγξτε το browser console για JavaScript errors
- Δοκιμάστε διαφορετικό browser

## 📊 Παράδειγμα Workflow Χρήσης

```bash
# 1. Εκπαίδευση μοντέλου
python mountain_car_ppo.py

# 2. Δημιουργία visualizations (αυτόματα μετά την εκπαίδευση, ή εκτελέστε χειροκίνητα)
python visualize_training.py

# 3. Άνοιγμα main dashboard στον browser
# Windows: start visualizations/main_dashboard.html
# Mac: open visualizations/main_dashboard.html
# Linux: xdg-open visualizations/main_dashboard.html

# 4. Παρακολούθηση παιχνιδιού πράκτορα
python view_agent.py
```

Απολαύστε την εξερεύνηση των αποτελεσμάτων εκπαίδευσης Mountain Car PPO! 🚗⛰️

