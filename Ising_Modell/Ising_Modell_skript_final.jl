using ITensors
using LinearAlgebra
using Random
using DelimitedFiles
using Printf

#Parameter festlegen:
tau= 0.1 #lokaler Zeitschritt des TEBD-Algorithmus
t = 50  #Gesamtzeit um die der Anfangsstring S_0 entwickelt werden soll
L = 60  #Länge der Kette
pos0 = L÷2 #Position auf die der Anfangsstring S_0 lokalisiert ist
bond_dim = 300 #maximale Bond Dimension

#Parameter des Ising Modells
J=1 
hz=0.8
hx= [s for s in 0.:0.2:1]

#Name des Ordners wo die Daten abgespeichert werden
output_folder = "chaotisches_Ising_Modell_300_tau_0.1"
#erzeugt Ordner mit dem Namen "output_folder"
mkpath(output_folder)


#Erzeugt die Lagen im TEBD Algortihmus:

#=Die Funktion bekommt ein Vektor sites bestehend aus ITensor Indizes,
tau entspricht dem lokalen Zeitschritt des TEBD Algorithmus, 
odd ist ein Boolean welcher bestimmt ob die Lage generiert durch H_ungerade oder die Lage aus H_gerade erzeugt werden soll,
L ist die Systemgröße welche o.B.d.A. gerade ist, 
(J,hx,hz) sind die Parameter im Ising Modell
=#
function U_layers(sites, tau, odd, L, J, hx, hz)
    
    #Array U_alpha initialisieren in dem sich die Tensoren von Rang 4 befinden aus dem unser Schaltkreis aufgebaut ist:
    U_alpha = ITensor[]

    
    
    #Summe über ungerade Indizes (H_ungerade):
    if odd == true
        
        #Über alle Gitterpunkte iterieren
        for j in 1:2:(L - 1)
            s1 = sites[j]
            s2 = sites[j + 1]
            #Auftretende Operatoren an den richtigen Stellen im Gitter definieren
            Sx1 = op("Sx", s1)
            Sx2 = op("Sx", s2)
            Sz1 = op("Sz", s1)
            Sz2 = op("Sz", s2)
            Id1 = op("Id", s1)
            Id2 = op("Id", s2)
            
            #Trotter-Suzuki-Zerlegung:
            
            #Randterm addieren (für j=1 müssen die Randterme berücksichtigt werden):
            if j == 1
                hj =
                J * Sx1 * Sx2 +
                hx / 2 * (Sx1 * Id2 + Id1 * Sx2) +
                hz / 2 *(Sz1 * Id2 + Id1 * Sz2) +
                hx/2 * Sx1 * Id2 + hz/2 * Sz1 * Id2 #hier wird Randterm addiert
                #Gj sind die Tensoren aus denen unser Schaltkreis besteht
                Gj = exp(-im * tau * hj)
                push!(U_alpha,Gj)
                continue
            end
            
            #die kummutierenden Summanden h_{i,i+1} initialisieren
            hj =
                J * Sx1 * Sx2 +
                hx / 2 * (Sx1 * Id2 + Id1 * Sx2) +
                hz / 2 *(Sz1 * Id2 + Id1 * Sz2) 
            Gj = exp(-im * tau * hj)
            push!(U_alpha,Gj)
        end
    #Summe über gerade Indices (H_gerade):
    else
        for j in 2:2:(L - 2)
            s1 = sites[j]
            s2 = sites[j + 1]
            #Auftretende Operatoren an den richtigen Stellen im Gitter definieren
            Sx1 = op("Sx", s1)
            Sx2 = op("Sx", s2)
            Sz1 = op("Sz", s1)
            Sz2 = op("Sz", s2)
            Id1 = op("Id", s1)
            Id2 = op("Id", s2)
            #die kummutierenden Summanden h_{i,i+1} initialisieren
            hj =
                J * Sx1 * Sx2 +
                hx / 2 * (Sx1 * Id2 + Id1 * Sx2) +
                hz / 2 *( Sz1 * Id2 + Id1 * Sz2)
            #Gj sind die Tensoren aus denen unser Schaltkreis besteht
            Gj = exp(-im * tau * hj)
            push!(U_alpha,Gj)
        end
        #Randterm hinzufügen
        Gj = exp(-im * tau * (hx/2 * op("Sx", sites[L]) + hz/2 * op("Sz", sites[L])))
        push!(U_alpha,Gj)
    end

    
    return U_alpha
end



#Zur Bestimmung eines Zeitschritts t werden iterativ mit der Funktion TEBD_single_timestep t/tau lokale Zeitschritte tau durchgeührt: 

#=
S0 ist ein MPO der um einen Zeitschritt entwickelt werten soll
W_odd bzw. W_even sind die Lagen die in Sandwitch-Form auf den MPO wirken, vergleiche Abb. 2.4 der Bachelorarbeit 
tau ist der lokale Zeitschritt im TEBD Algorithmus
L ist die Systemgröße
(J,hx,hz) sind die Parameter des Ising Hamiltonians
bond_dim ist die maiximal erlaubte Bond Dimension
=#
function TEBD_single_timestep(S0, W_odd, W_even, tau, L, J, hx, hz, bond_dim)
    
    #Nur ein Zeitschritt:
    t=1 
    sites = siteinds(S0, plev = 0)
    tsteps = ceil(t/tau)
    S = copy(S0)
    L = length(S)
   
    
    #TEBD2 Algorithmus anwenden mit den Trotter-Gatter die mit U_layers erstellt werden: 
    for n in 1:tsteps
        
    
        #Zuerst wird die Lage odd, danach die Lage even und dann wieder odd angewendet (vergleiche  Abb. 2.4), dabei führt
        #die Funktion apply eine Kompression der Bond Dimension durch um den neuen Operator wieder als MPO darstellen zu können
        S = apply(W_odd, S; maxdim = bond_dim, apply_dag=true)
        S = apply(W_even, S; maxdim = bond_dim, apply_dag=true) 
        S = apply(W_odd, S; maxdim = bond_dim, apply_dag=true) 
        
#         normalize!(S)
        
    end
    
    normalize!(S)
    
    return S
end


#Funktion berechnet das totale Gewicht aller Paulistrings mit rechten Endpunkt x_r in der Basisentwicklung von S(t):
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
    #RHO_R ist die Variable in der unser Ergebnis für die Operatordichte rho_R(s,t) abgespeichert wird
    RHO_R = 0
    for m in projection
       
        #Pauli Matrizen als Tensor an der Stelle x_r
        proj_op = op(m, sites[x_r])
        #Berechne die Projektion auf einen der drei Pauli-Matrizen (teile durch sqrt(2) zur Normierung)
        rho_xyz = R * proj_op * 1/sqrt(2) 
        #addiere die Projektionen auf die drei Pauli-Matrizen zusammen
        RHO_R += scalar(rho_xyz * dag(rho_xyz))
        
    end

    
    return abs(RHO_R) 
    
end


#Funktion berechnet das totale Gewicht aller Paulistrings mit linken Endpunkt x_l in der Basisentwicklung von S(t):
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
    #RHO_L ist die Variable in der unser Ergebnis für die Operatordichte rho_L(s,t) abgespeichert wird
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

#Erzeuge die Zeitentwicklung unter des Ising Hamiltonians und berechnet die beiden Operatordichten für alle Zeiten und Positionen:

#=
S0 ist ein MPO der um t Zeiten entwickelt werden soll
tau ist der lokale Zeitschritt im TEBD Algorithmus
t ist die gesamte Simulationszeit
(J,hx,hz) sind die Parameter des Ising Hamiltonians
bond_dim ist die maiximal erlaubte Bond Dimension
=#
function generate_data(S_0, tau, t, J, hx, hz,bond_dim)
    
    sites = siteinds(S_0, plev = 0)
    L = length(S_0)
    
    #Berechne die Lagen im TEBD Algorithmus mit der Funktion U_layers
    W_odd = U_layers(sites, tau/2, true, L, J, hx, hz)
    W_even = U_layers(sites, tau, false, L, J, hx, hz)
    
    #Anfangsstring normieren
    S_prev = copy(S_0) 
    S_prev ./= sqrt(2)

    #Arrays initialisiern um die Operatordichten zu speichern
    RHO_R = zeros(t+1,L)
    RHO_L = zeros(t+1,L)

    #Operatordichten des Produktzustands S_prev (bei t=0) speichern:
    for s in 1:L
        RHO_R[t+1,s] = rho_R(S_prev,s)
        RHO_L[t+1,s] = rho_L(S_prev,s)
    end
    
    #Iteriere über alle Zeiten
    for n in 1:t
        
        #Der Anfangsstring wird durch die Funktion TEBD_single_timestep um einen Zeitschritt entwickelt und bekommt den MPO vom 
        #vorherigen Zeitschritt wieder übergeben
        S_new = TEBD_single_timestep(S_prev, W_odd, W_even, tau, L, J, hx, hz, bond_dim)
        
        #berechne für den Zeitpunkt t die Operatordichten rho_L(x,t) und rho_R(x,t) an allen Gitterpunkten
        for s in 1:L

            RHO_R[t+1-n,s] = rho_R(S_new,s)
            RHO_L[t+1-n,s] = rho_L(S_new,s) 
            
        end
        #Setze S_prev auf den neuen Zeitentwickelten MPO S_new um in der nächsten Iteration den nächsten Zeitschritt zu erhalten
        S_prev = S_new
        
    end
    
    
    return RHO_L, RHO_R 
end


#Simulation durchführen und Daten im Ordner "output_folder" speichern:

#Hilbertraum des gesamten Systems definieren
sites = siteinds("S=1/2",L)
#Anfangs Paulistring definieren
Config = repeat(["Id"],L)
Config[pos0] = "Z" #Lokalisiert bei pos0 mit der Pauli-Matrix Z
Sz = MPO(sites, Config)
S0 = copy(Sz) 
    
#Führe für jeden longitudinale Komponente hx eine Simulation durch
for j in hx
    t1 = time() #zur Zeitmessung
    #die Operatordichten für alle Zeiten und Positionen erzeugen
    RHO_L, RHO_R = generate_data(S0, tau, t, J, j, hz, bond_dim)

    #Zur besseren Handhabung der Daten
    RHO_L[abs.(RHO_L) .< 1E-16] .= 0
    RHO_R[abs.(RHO_R) .< 1E-16] .= 0

    #Um die Daten den gewählten Parametern zuordnen zu können
    filename = "J_"  * string(J) * "_hx_" * string(j) * "_hz_" * string(hz) * "_result.txt"
    output_file = joinpath(output_folder, filename) 
    t2 = time()
    @printf("Needed time[s]: %0.3f\n", t2 - t1)

    #Fülle die .txt Datei "filename" mit den Operatordichten und speicher sie in "output_folder"
    open(output_file, "w") do io
            writedlm(io, [RHO_L RHO_R])
    end
end