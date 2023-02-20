<h1 style="font-weight:bold; text-align: center; margin: 0px; font-size: 30px; padding:0px;">Bachelorarbeit: Operatorausbreitung in zufälligen Quantenschaltkreisen</h1>
Auf dieser Webseite ist der gesamte Code zu finden, welcher für die Bachelorarbeit geschrieben und verwendet wurden. Alle generierten Daten die mit dem Code erzeugt wurden sind hier zu finden.

Der Ordner **Random_Circuit_Model** beinhaltet folgende Datein:
- Analytische_Ergebnisse_final.ipynb, ist das Jupyter Notebook in dem die Abbildungen für Abschnitt 5.2 der Bachelorarbeit erzeugt wurden.
- Random_circuit_model_final.ipynb, ist das Jupyter Notebook in dem der Code zur Simulation der Operatorausbreitung im zufälligen Quantenschaltkreis geschrieben wurde. Das Notebook enthält eine detaillierte Beschreibung des Codes.
- Random_Circuit_Model_skript_final.jl, ist ein Julia Skript, welches in der Art auf dem PGI8Cluster laufen gelassen wurde, um mehrere Durchläufe des zufälligen Quantenschaltkreises zu erzeugen. Das Skript enthält den selben Code wie das Notebook Random_circuit_model_final.ipynb, ist aber weniger detailliert beschrieben.
- Die Ordner "results_50" bis "results_1500" umfassen alle Daten, die mit dem Skript Random_Circuit_Model_skript_final.jl auf dem PGI8Cluster erzeugt wurden. Dabei entspricht die Zahl im Namen des Ordners, der gewählten Bond Dimension. Die Ordner selbst beinhalten 100 Durchläufe des zufälligen Quantenschaltkreises, als .txt Dateien abgespeichert. Der Inhalt der .txt Dateien sind die Operatordichten $\rho_{L}(s,\tau)$ und $\rho_{R}(s,\tau)$ für alle Gitterpunkte $s$ und alle Zeiten $\tau$. Die .txt Dateien sind für jeden Durchlauf mit den verwendeten Seed beschriftet, um die Daten reproduzieren zu können.
- Random_Circuit_Model_Auswertung_final.ipynb, beinhaltet die gesamte Auswertung der Daten, die in den Ordnern "results_50" bis "results_1500" gespeichert sind. Alle Abbildungen in Abschnitt 5.3 der Bachelorarbeit wurden in diesem Notebook erzeugt.

Der Ordner **Ising_Modell** beinhaltet folgende Datein:
- Ising_Modell_final.ipynb, ist das Jupyter Notebook in dem der Code zur Simulation der Operatorausbreitung für das Ising Modell mit transversalem und longitudinalem Feld geschrieben wurde. Das Notebook enthält eine detaillierte Beschreibung des Codes.
- Ising_Modell_skript_final.jl, ist ein Julia Skript, welches in der Art auf dem PGI8Cluster laufen gelassen wurde. Für eine bestimmte Bond Dimension und Parameter $(J,h_{x},h_{z})$ wurde mit dieser Datei die Operatorausbreitung simuliert und die Operatordichten berechnet. Das Skript enthält den selben Code wie das Notebook Ising_Modell_final.ipynb, ist aber weniger detailliert beschrieben.
- Die Ordner "chaotisches_Ising_Modell_50_delta_0.1" bis "chaotisches_Ising_Modell_300_delta_0.1" umfassen die Daten, die mit dem Skript Random_Circuit_Model_skript_final.jl auf dem PGI8Cluster erzeugt wurden. Die erste Zahl im Namen des Ordners entspricht der Bond Dimension mit der die Daten generiert wurden. Es wurden Daten für Bond Dimensionen 50, 100, 150, 200 und 300 erzeugt. Die zweite Zahl entspricht dem gewählten lokalen Zeitschritt $\delta$ im TEBD-Algorithmus, welcher für alle Simulationen auf $0.1$ gesetzt wurde. Die Ordner selbst beinhalten .txt Datein. Der Inhalt der .txt Dateien sind die Operatordichten $\rho_{L}(s,\tau)$ und $\rho_{R}(s,\tau)$ für alle Gitterpunkte $s$ und alle Zeiten $\tau$. Die Beschriftung der .txt Datein ist so gewählt, dass die erste Zahl Parameter J entspricht, die zweite Zahl Parameter $h_{x}$ und die dritte Zahl Parameter $h_{z}$.
- Ising_Modell_Auswertung_final.ipynb, beinhaltet die gesamte Auswertung der Daten, die in den Ordnern "chaotisches_Ising_Modell_50_delta_0.1" bis "chaotisches_Ising_Modell_300_delta_0.1" gespeichert sind. Alle Abbildungen in Kapitel 6 der Bachelorarbeit wurden in diesem Notebook erzeugt.
