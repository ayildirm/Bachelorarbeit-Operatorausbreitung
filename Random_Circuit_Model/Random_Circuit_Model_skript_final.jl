using ITensors
using LinearAlgebra
using Random
using DelimitedFiles
using Printf

#Parameter festlegen:

t = 16  #Gesamte Zeitdauer um die der Anfangsstring S_0 entwickelt werden soll
L = 41  #Länge der Kette
pos0 = (L÷2) + 1 #Position auf die der Anfangsstring S_0 lokalisiert ist
N = 3100 #Für den Seed (siehe paar Zeilen weiter unten)
bond_dim = 1500 #maximale Bond Dimension

#Name des Ordners wo die Durchläufe gespeichert werden sollen
output_folder = "results_1500"
#Erzeugt einen Ordner mit dem Namen "output_folder"
mkpath(output_folder)
#Array mit den Zahlen die als Seed verwendet werden
my_seeds = [s for s in 3091:N]


#Zur Generierung Haar-verteilter Tensoren von Rang 4 und Rang 2
function generate_gate(idx1, idx...; elT = ComplexF64)
    
    #Gesamtanzahl der Indizes bzw. Rang des Haar-verteilten Tensors der generiert werden soll
    N_legs = length(idx) + 1
    #Dimension des ersten Indizes
    hilb = dim(idx1)
    
    #Hilberträume müssen auf allen sites gleich sein:
    @assert all(i -> dim(i) == hilb, idx)
    
    #Implementierung des Algorithmus zur Generierung Haar-verteilter Matrizen:

    #Initialisiere quadratische Matrix A der richtigen Größe mit komplexen standardnormalverteilten Einträgen
    A = randn(elT, hilb^N_legs, hilb^N_legs)

    #QR-Zerlegung der Matrix A
    Q,R = qr(A)

    #Die Diagonalmatrix signR bzw. im Haupttext \Lambda ist so konstruiert, dass ihre Einträge der Phase der Matrix R entsprechen
    signR = diagm(sign.(real(diag(R))))

    #Multipliziere signR mit der unitären Matrix Q um eine Haar-verteilte Matrix zu erhalten
    Q = signR*Q
    
    #Erstelle mit der Haar-veteilten Matrix ein ITensor mit der richtigen Anzahl an physikalischen Indices, vgl. Schaltbild
    #(ein 1-Qubit-Gatter entspricht einer 2x2 Matrix mit 2 physiklaischen Indices, 
    #ein 2-Qubit-Gatter entspricht einer 4x4 Matrix mit 4 physiklaischen Indices, etc.)

    #Hilfsvariable um Matrix in ein ITensor umzuwandeln
    hilbs = repeat([hilb], N_legs)
    #Wandle die Matrix in einen ITensor um mit der entsprechenden Anzahl an Indizes und Zuordnung der Indizes
    return ITensor(reshape(Q, hilbs...,hilbs...), idx1, idx..., prime(idx1), prime.(idx)...)
end

#Generiert eine Lage zum Zeitpunkt t bestehend aus Haar-verteilten Gattern entsprechend dem Schaltkreis Modell
function Layer(t,L,sites)
    
    #Circuit mit gerader Anzahl an sites (L gerade)
    if L%2 == 0
        #even layer (t gerade):
        if t%2 == 0
            #Initialisiere Array um die Lage V gefüllt mit Haar-verteilten Tensoren zu speichern
            W = Vector{ITensor}(undef, (L÷2)+1)
            #Tensoren von Rang 2 am Rand, d.h. generate_gate bekommt nur ein Index als Input
            W[1] = generate_gate(sites[1])
            W[end] = generate_gate(sites[L])

            #Alle Tensoren von Rang 4 generieren
            for j in 2:L÷2

                #generate_gate bekommt die beiden Indizes  als Input, die zum Zeitpunkt t interagieren
                W[j] = generate_gate(sites[2*(j-1)], sites[2*j - 1])
            end

        #odd layer (t ungerade):
        else
            #Initialisiere Array um die Lage W gefüllt mit Haar-verteilten Tensoren zu speichern
            W = Vector{ITensor}(undef, L÷2)

            #Bei einem Schaltkreis mit L gerade, besteht die Lage für t ungerade nur aus Tensoren von Rang 4
            for j in 1:L÷2
                W[j] = generate_gate(sites[2*j - 1], sites[2*j])
            end

        end
    #Circuit mit ungerader Anzahl an Gitterpunkte (L ungerade)
    else
        #Initialisiere Array um die Lage W gefüllt mit Haar-verteilten Tensoren zu speichern
        #Für L ungerade bestehen die Lagen für t gerade bzw. ungerade t aus gleich vielen Tensoren 
        W = Vector{ITensor}(undef, ((L-1)÷2)+1)
        #even layer (t gerade):
        if t%2 == 0
            #Tensoren von Rang 2 am Rand
            W[1] = generate_gate(sites[1])

            for j in 2:(L-1)÷2 + 1

                #Alle anderen Tensoren sind von Rang 4
                W[j] = generate_gate(sites[2*(j-1)], sites[2*j - 1])
            end

        #odd layer (tau ungerade):
        else
            #Tensoren von Rang 2 am Rand
            W[end] = generate_gate(sites[L])
            for j in 1:(L-1)÷2
                #Alle anderen Tensoren sind von Rang 4
                W[j] = generate_gate(sites[2*j - 1], sites[2*j])
            end

        end
    end
        

    return W
end

#Die Zeitentwicklung nach einem Zeitschritt im Heisenbergbild
function single_timestep(S0, t, bond_dim)

    #Indizes des MPOs S0
    sites = siteinds(S0, plev = 0)

    S = copy(S0)
    L = length(S)
    
    #Generiere Lage Haar-verteilter Tensoren zum Zeitpunkt t
    W = Layer(t, L, sites)

    #Heisenberg'sche Zeitentwicklung:
    S = apply(W, S; maxdim = bond_dim, apply_dag=true)
    normalize!(S)

    return S
end

#Funktion berechnet das totale Gewicht aller Paulistrings mit rechten Endpunkt x_r in der Basisentwicklung von S(\tau):

function rho_R(S,x_r)
    
    sites = siteinds(S, plev = 0)
    Sc = copy(S)
    
    #Bringe den MPO Sc in gemischt-kanonische Form mit Orthogonalisierungszentrum bei x_r, damit die Kontraktion der Tensoren 
    #links von x_r trivial wird, vergleiche Bachelorarbeit:
    orthogonalize!(Sc, x_r)
    L = length(S)
    
    # 1. Berechne den in der Bachelorarbeit genannte Tensor T, indem alle Tensoren rechts von x_r auf die Identität projiziert
    #werden und kontrahiere über alle Bond Indizes in T:
    
    T = ITensor(1)
    for i in L:-1:x_r+1
        #Die i-te Stelle des MPOs Sc wird auf die Identität projiziert
        R = Sc[i]
        #Definiere den Identitäts Tensor an der Stelle i
        id = op("Id", sites[i])
        #projiziere auf die Identität und kontrahiere die Bond Indizes (teile durch sqrt(2) zur Normierung)
        T *= (R * id) * 1/sqrt(2)
    end
    
    # 2. Projiziere Gitterpunkt x_r auf die Pauli Matrizen X, Y und Z:
    
    #Multipliziere die Stelle x_r im MPO mit T
    O = Sc[x_r]
    R = O*T
    
    #Wir projizieren auf die drei Pauli-Matrizen
    projection = ["X", "Y", "Z"]
    #RHO_R ist die Variable in der unser Ergebnis für die Operatordichte rho_R(s,\tau) abgespeichert wird
    RHO_R = 0
    
    #Über alle Pauli-Matrizen iterieren
    for m in projection
       
        #Pauli-Matrizen als Tensor an der Stelle x_r
        proj_op = op(m, sites[x_r])
        #Berechne die Projektion auf einen der drei Pauli-Matrizen (teile durch sqrt(2) zur Normierung)
        rho_xyz = R * proj_op * 1/sqrt(2) 
        #Verknüfe rho_xyz mit sich selber und addiere die Projektionen auf die drei Pauli-Matrizen zusammen
        RHO_R += scalar(rho_xyz * dag(rho_xyz))
        
    end

    
    return abs(RHO_R) 
    
end

#Funktion berechnet das totale Gewicht aller Paulistrings mit linken Endpunkt x_l in der Basisentwicklung von S(\tau):
function rho_L(S,x_l)
    
    sites = siteinds(S, plev = 0)
    Sc = copy(S)
    
    #Bringe den MPO Sc in gemischt-kanonische Form mit Orthogonalisierungszentrum bei x_l, damit die Kontraktion der Tensoren 
    #rechts von x_l trivial wird:
    orthogonalize!(Sc, x_l)
    L = length(S)
    
    # 1. Berechne den Tensor T indem wir die sites 1:x_l-1 auf die Identität projizierne und
    # kontrahiere über alle Bond Indices in T:
    
    T = ITensor(1)
    for i in 1:x_l-1
        #Die i-te Stelle des MPOs Sc wird auf die Identität projiziert
        R = Sc[i]
        #Definiere den Identitäts Tensor an der Stelle i
        id = op("Id", sites[i])
        #projiziere auf die Identität und kontrahiere die Bond Indizes (teile durch sqrt(2) zur Normierung)
        T *= (R * id) * 1/sqrt(2)
    end
    
    # 2. Projiziere site x_l auf die Pauli-Matrizen X, Y und Z
    
    #Multipliziere die Stelle x_l im MPO Sc mit T
    O = Sc[x_l]
    R = O*T
    
    #Wir projizieren auf die drei Pauli-Matrizen
    projection = ["X", "Y", "Z"]
    #RHO_L ist die Variable in der unser Ergebnis für die Operatordichte rho_L(s,\tau) abgespeichert wird
    RHO_L = 0
    for m in projection
       
        #Pauli Matrizen als Tensor an der Stelle x_l
        proj_op = op(m, sites[x_l])
        #Berechne die Projektion auf einen der drei Pauli-Matrizen (teile durch sqrt(2) zur Normierung)
        rho_xyz = R * proj_op * 1/sqrt(2)
        #addiere die Projektionen auf die drei Pauli-Matrizen zusammen
        RHO_L += scalar(rho_xyz * dag(rho_xyz))
        
    end

    
    return abs(RHO_L) 
    
end

#Erzeugt ein Durchlauf und speichert die Daten für die Operatordichten in RHO_L bzw. RHO_R
function generate_data(seed, S_0, t, bond_dim)
    
    #Anfangsstring normieren
    S_prev = copy(S_0) 
    S_prev ./= sqrt(2)

    L = length(S_0)
    #Arrays in denen die Operatordichten für alle Zeiten \tau und Positionen s abgespeichert werden
    RHO_R = zeros(t+1,L)
    RHO_L = zeros(t+1,L)

    #Rechter Endpunkt des Produktoperators (bei \tau = 0) S_prev speichern
    for s in 1:L
        RHO_R[t+1,s] = rho_R(S_prev,s)
        RHO_L[t+1,s] = rho_L(S_prev,s)
    end
    
    #Seed setzen zur Reproduzierbarkeit
    Random.seed!(seed)
    #Iteriere über alle Zeiten
    for n in 1:t
        #Der Anfangsstring wird durch die Funktion single_timestep um einen Zeitschritt entwickelt und bekommt den MPO vom 
        #vorherigen Zeitschritt wieder übergeben
        S_new = single_timestep(S_prev, n, bond_dim)
        
        #berechne für den Zeitpunkt t die Operatordichten rho_L(s,\tau) und rho_R(s,\tau) an allen Gitterpunkten
        for s in 1:L

            RHO_R[t+1-n,s] = rho_R(S_new,s)
            RHO_L[t+1-n,s] = rho_L(S_new,s) 
            
        end
        #Setze S_prev auf den neuen Zeitentwickelten MPO S_new um in der nächsten Iteration den nächsten Zeitschritt zu erhalten
        S_prev = S_new
        
    end
    
    
    return RHO_L, RHO_R 
end

#Simulation durchführen und Daten im Ordner "output_folder" speichern

#compiling run (dient der schnelleren Ausführung des ersten Durchlaufs)
begin# t = 2, L = 2, pos0 = 1, bond_dim = 2, seed = 1234
    sites = siteinds("S=1/2",4)
    #Anfangs Paulistring definieren
    Config = repeat(["Id"],4)
    Config[2] = "Z"
    Sz = MPO(sites, Config)
    S0 = copy(Sz) 
    generate_data(1234, S0, 4, 2) 
end


#Hilbertraum des gesamten Systems definieren:
sites = siteinds("S=1/2",L)
#Anfangs Paulistring definieren:
Config = repeat(["Id"],L)
Config[pos0] = "Z"
Sz = MPO(sites, Config)
S0 = copy(Sz) 
    

#Führe für jeden Seed einen Durchlauf des zufälligen Quantenschaltkreis durch
for seed in my_seeds
    t1 = time() #zur Zeitmessung
    #Die Operatordichten für alle Zeiten und Positionen erzeugen
    RHO_L, RHO_R = generate_data(seed, S0, t, bond_dim)

    #Zur besseren Handhabung der Daten
    RHO_L[abs.(RHO_L) .< 1E-16] .= 0
    RHO_R[abs.(RHO_R) .< 1E-16] .= 0

    #Um die Daten den Seeds zuordenen zu können
    filename = string(seed) * "_result.txt"
    output_file = joinpath(output_folder, filename) 
    t2 = time()
    @printf("Used Seed: %i, needed time[s]: %0.3f\n", seed, t2 - t1)

    #Fülle die .txt Datei "filename" mit den Operatordichten und speicher sie in "output_folder"
    open(output_file, "w") do io
            writedlm(io, [RHO_L RHO_R])
    end
end

